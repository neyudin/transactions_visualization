import numpy as np
import pandas as pd
import os
import sys
import re
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import dash
import dash_core_components as dcc
import dash_html_components as html
import base64
import datetime
import io
from dash.dependencies import Input, Output

app = dash.Dash()

app.scripts.config.serve_locally = True

app.layout = html.Div([
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Поднесите или ',
            html.A('выберите файл')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=True
    ),
    html.Div(id='output-image-upload')
])

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            customer_df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            customer_df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    
    num_topics = 0

    for col in customer_df.columns:
        m = re.findall('(?<=t_)\d+', col)
        if len(m) == 1:
            cur_num = int(m[0]) + 1
            if cur_num >= num_topics:
                num_topics = cur_num
            
    words = customer_df.category.unique()
    num_words = words.shape[0]

    local_phi = pd.DataFrame(dict([('topic_%i'%i, np.zeros(num_words)) for i in range(num_topics)]), index=words)
    local_tdw = pd.DataFrame(dict([(word, np.zeros(num_topics)) for word in words]),
                             index=['topic_%i'%i for i in range(num_topics)])
    local_theta = np.zeros(num_topics)

    for i in range(num_topics):
        local_theta[i] = customer_df.loc[0, 't_%i|d'%i]    

    for row_idx in customer_df.index:
        for i in range(num_topics):
            local_tdw.loc['topic_%i'%i, customer_df.loc[row_idx, 'category']] = customer_df.loc[row_idx, 't_%i|dw'%i]
            local_phi.loc[customer_df.loc[row_idx, 'category'], 'topic_%i'%i] = customer_df.loc[row_idx, 'w|t_%i'%i]

    wd_df = pd.DataFrame({'w|d': (local_phi.as_matrix() * local_theta.reshape(1, -1)).sum(axis=1)},
                         index=local_phi.index)

    local_theta_df = pd.DataFrame({'t|d': local_theta}, index=local_tdw.index)

    def temporal_tdw(df):
        df_dict = dict(('t_%i|dW'%i, [0.0]) for i in range(num_topics))
        df_dict['step'] = df.step.unique()[0]
        df_dict['category'] = ""
        df_dict['fraud'] = ""
        df_dict['merchant'] = ""
        df_dict['age'] = ""
        df_dict['gender'] = ""
        df_index = df.index
        unique_category, unique_index = np.unique(df.category, return_index=True)
        if (unique_category.shape[0] > 1) and (df.shape[0] > 1):
            for unique_idx in unique_index:
                for i in range(num_topics):
                    df_dict['t_%i|dW'%i][0] += df.loc[df_index[unique_idx], 't_%i|dw'%i] *\
                                               wd_df.loc[df.loc[df_index[unique_idx], 'category'], 'w|d']
                df_dict['category'] += df.loc[df_index[unique_idx], 'category'] + ";"
            norm = 0.0
            for i in range(num_topics):
                norm += df_dict['t_%i|dW'%i][0]
            for i in range(num_topics):
                df_dict['t_%i|dW'%i][0] /= norm
        else:
            for i in range(num_topics):
                df_dict['t_%i|dW'%i][0] = df.loc[df_index[0], 't_%i|dw'%i]
            df_dict['category'] += df.loc[df_index[0], 'category']
        df_dict['amount'] = df.amount.sum()
        for row_idx in df.index:
            df_dict['fraud'] += str(df.loc[row_idx, 'fraud'])
            df_dict['merchant'] += df.loc[row_idx, 'merchant'] + "; "
        df_dict['age'] = df.loc[df_index[0], 'age']
        df_dict['gender'] = df.loc[df_index[0], 'gender']
        return pd.DataFrame(df_dict, index=[df_index[0]])

    tdW_df = customer_df.groupby('step').apply(temporal_tdw).reset_index(drop=True)

    seed = 417
    np.random.seed(seed=seed)
    topic_colors = np.random.choice(a=np.hstack((np.arange(1, num_topics + 1),
                                                 np.arange(1, num_topics + 1))), size=(num_topics, 2), replace=False)
    topic_colors = (np.hstack((topic_colors, np.arange(num_topics).reshape(-1, 1))) /\
                     num_topics * 255).astype(np.int64)

    words_colors = (local_phi.as_matrix() / local_phi.sum(axis=1).as_matrix().reshape(-1, 1)).dot(\
                    topic_colors).astype(np.int64)

    words_colors_df = pd.DataFrame({'r': words_colors[:, 0],
                                    'g': words_colors[:, 1],
                                    'b': words_colors[:, 2]}, index=local_phi.index)

    topic_colors_df = pd.DataFrame({'r': topic_colors[:, 0],
                                    'g': topic_colors[:, 1],
                                    'b': topic_colors[:, 2]}, index=local_tdw.index)

    go_x_vals_wt = {}
    for word in words:
        go_x_vals_wt[word] = []

    for row_idx in customer_df.index:
        go_x_vals_wt[customer_df.loc[row_idx, 'category']].append(customer_df.loc[row_idx, 'step'])

    for word in words:
        go_x_vals_wt[word] = np.unique(go_x_vals_wt[word])

    go_y_vals_wt = {}
    go_y_axis = wd_df.sort_values(by='w|d', ascending=False).index
    for number, word in enumerate(go_y_axis):
        go_y_vals_wt[word] = number + 1

    go_text_wt = {}

    for word in words:
        go_text_wt[word] = []

    for row_idx in tdW_df.index:
        go_text_wt[tdW_df.loc[row_idx, 'category']].append(\
        "amount: %.02f<br>fraud: %s<br>merchant: %s"%(tdW_df.loc[row_idx, 'amount'],
                                                      tdW_df.loc[row_idx, 'fraud'],
                                                      tdW_df.loc[row_idx, 'merchant']))

    trace_arr = []

    for word in words:
        trace_arr.append(go.Scatter(
                                    x=go_x_vals_wt[word],
                                    y=[go_y_vals_wt[word]] * len(go_x_vals_wt[word]),
                                    text=go_text_wt[word],
                                    mode='markers',
                                    marker=dict(color=('rgb(%i, %i, %i)'%(words_colors_df.loc[word, 'r'],
                                                                          words_colors_df.loc[word, 'g'],
                                                                          words_colors_df.loc[word, 'b']
                                                                         )
                                                      ),
                                                size=18
                                               ),
                                    name=word
                                   )
                        )

    topic_z_arr = []

    for i in range(num_topics):
        topic_z_arr.append(list(tdW_df.loc[:, 't_%i|dW'%i]))

    go_text_tdW = []

    for i in range(num_topics):
        go_text_tdW.append([])
        for row_idx in tdW_df.index:
            go_text_tdW[i].append("age: %s<br>gender: %s<br>fraud: %s"%(tdW_df.loc[row_idx, 'age'],
                                                                        tdW_df.loc[row_idx, 'gender'],
                                                                        tdW_df.loc[row_idx, 'fraud']))

    trace_arr.append(go.Contour(z=topic_z_arr,
                                x=tdW_df.step.as_matrix(),
                                y=np.arange(num_topics),
                                text=go_text_tdW,
                                contours=dict(coloring='heatmap'),
                                yaxis='y2',
                                showscale=False,
                                name='Распределение тем во времени'
                               )
                    )

    trace_arr.append(go.Scatter(name='is fraud',
                                yaxis='y3',
                                mode='markers',
                                x=tdW_df.step.as_matrix(),
                                y=tdW_df.fraud.as_matrix(),
                                text=['is fraud'] * tdW_df.step.as_matrix().shape[0],
                                marker=dict(color=['rgb(%i, 0, 0)'%(255 * min(1, int(i))) for i in tdW_df.fraud],
                                            size=18
                                           )
                               )
                    )

    trace_arr.append(go.Bar(name='Распределение тем',
                            x=['topic_%i'%i for i in range(num_topics)],
                            y=[local_theta_df.loc['topic_%i'%i, 't|d'] for i in range(num_topics)],
                            xaxis='x2',
                            yaxis='y4',
                            marker=dict(color=['rgb(%i, %i, %i)'%(topic_colors_df.loc['topic_%i'%i, 'r'],
                                                                  topic_colors_df.loc['topic_%i'%i, 'g'],
                                                                  topic_colors_df.loc['topic_%i'%i, 'b'])
                                              for i in range(num_topics)])))

    trace_arr.append(go.Bar(name='Распределение категорий',
                            x=[word for word in words],
                            y=[wd_df.loc[word, 'w|d'] for word in words],
                            xaxis='x3',
                            yaxis='y5',
                            marker=dict(color=['rgb(%i, %i, %i)'%(words_colors_df.loc[word, 'r'],
                                                                  words_colors_df.loc[word, 'g'],
                                                                  words_colors_df.loc[word, 'b'])
                                                                            for word in words])))

    shared_layout = dict(width = 1250,#5000,
                         height = 800,
                         title = 'Транзакции клиента d = %s'%customer_df.customer[0],
                         titlefont = dict(size = 36,
                                          family = "Verdana",
                                          color = "black"
                                         ),
                         margin = dict(l = 145),
                         showlegend = False,
                         xaxis = dict(domain = [0, 1],
                                      title = "<b>Отсчёты в днях<b>",
                                      titlefont = dict(size = 14,
                                                       family = "Verdana",
                                                       color = "black"
                                                      ),
                                      anchor = "y3",
                                      linecolor = 'black',
                                      linewidth = 2,
                                      mirror = True,
                                      zeroline = True,
                                      showline = True,
                                      zerolinewidth = 4
                                     ),
                         yaxis = dict(domain = [0.65, 1],
                                      title = "<b>w<b>",
                                      titlefont = dict(size = 14,
                                                       family = "Verdana",
                                                       color = "black"
                                                      ),
                                      tickvals = np.arange(1, len(go_y_axis) + 1),
                                      ticktext = go_y_axis,
                                      scaleanchor = "x",
                                      linecolor = 'black',
                                      linewidth = 2,
                                      mirror = True,
                                      zeroline = True,
                                      showline = True,
                                      zerolinewidth = 4
                                     ),
                         yaxis2 = dict(domain = [0.3, 0.65],
                                       title = "<b>p(t|d, w)<b>",
                                       titlefont = dict(size = 14,
                                                        family = "Verdana",
                                                        color = "black"
                                                       ),
                                       tickvals = np.arange(num_topics),
                                       ticktext = ['topic_%i'%i for i in range(num_topics)],
                                       scaleanchor = "x",
                                       linecolor = 'black',
                                       linewidth = 2,
                                       mirror = True,
                                       zeroline = True,
                                       showline = True,
                                       zerolinewidth = 4
                                      ),
                         yaxis3 = dict(domain = [0.2, 0.3],
                                       title = "<b>Разметка<b>",
                                       titlefont = dict(size = 14,
                                                        family = "Verdana",
                                                        color = "black"
                                                       ),
                                       tickvals = [0, 1],
                                       scaleanchor = "x",
                                       linecolor = 'black',
                                       linewidth = 2,
                                       mirror = True,
                                       zeroline = True,
                                       showline = True,
                                       zerolinewidth = 4
                                      ),
                         xaxis2 = dict(domain = [0, 0.45],
                                       title = "<b>Темы<b>",
                                       titlefont = dict(size = 14,
                                                        family = "Verdana",
                                                        color = "black"
                                                       ),
                                       anchor = "y4",
                                       linecolor = 'black',
                                       linewidth = 2,
                                       mirror = True,
                                       zeroline = True,
                                       showline = True,
                                       zerolinewidth = 4
                                      ),
                         yaxis4 = dict(domain = [0, 0.1],
                                       title = "<b>p(t|d)<b>",
                                       titlefont = dict(size = 14,
                                                        family = "Verdana",
                                                        color = "black"
                                                       ),
                                       anchor = "x2",
                                       linecolor = 'black',
                                       linewidth = 2,
                                       mirror = True,
                                       zeroline = True,
                                       showline = True,
                                       zerolinewidth = 4
                                      ),
                         xaxis3 = dict(domain = [0.55, 1],
                                       title = "<b>Категории<b>",
                                       titlefont = dict(size = 14,
                                                        family = "Verdana",
                                                        color = "black"
                                                       ),
                                       anchor = "y5",
                                       linecolor = 'black',
                                       linewidth = 2,
                                       mirror = True,
                                       zeroline = True,
                                       showline = True,
                                       zerolinewidth = 4
                                      ),
                         yaxis5 = dict(domain = [0, 0.1],
                                       title = "<b>p(w|d)<b>",
                                       titlefont = dict(size = 14,
                                                        family = "Verdana",
                                                        color = "black"
                                                       ),
                                       anchor = "x3",
                                       linecolor = 'black',
                                       linewidth = 2,
                                       mirror = True,
                                       zeroline = True,
                                       showline = True,
                                       zerolinewidth = 4
                                      )
                        )
    return html.Div([
        dcc.Graph(
            id='topic_modeling',
            figure={
                'data': trace_arr,
                'layout': go.Layout(
                    **shared_layout
                )
            }
        )
    ])


@app.callback(Output('output-image-upload', 'children'),
              [Input('upload-image', 'contents'),
               Input('upload-image', 'filename'),
               Input('upload-image', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


if __name__ == '__main__':
    app.run_server()