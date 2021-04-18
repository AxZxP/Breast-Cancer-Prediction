import numpy as np
import seaborn as sns
import matplotlib
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import plotly.tools as tls
import plotly.figure_factory as ff
import plotly.offline as py
import pandas as pd
from itertools import cycle

from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve


def set_sns_format(width=15, height=6):
    sns.set_theme(palette='pastel', context='notebook', rc={'savefig.dpi': 300})
    matplotlib.rcParams['figure.figsize'] = (width, height)
    return None


def suplot_num_features(df: pd.DataFrame, label=None):
    if label:
        label = df[f'{label}']
        df = df.drop(label, axis=1)

    df_num = df.select_dtypes(exclude=['O', 'category'])

    if len(df_num.columns) > 9:
        raise ValueError('More than 9 subplots is not a good idea, please send a subset.')

    facets = ((1, 1), (1, 2), (1, 3),
              (2, 1), (2, 2), (2, 3),
              (3, 1), (3, 2), (3, 3))

    fig = make_subplots(rows=3, cols=3,
                        subplot_titles=tuple(title.replace('_', ' ').title() for title in df_num.columns))

    for feature, idx in zip(df_num.columns, facets):
        order = dict(row=idx[0], col=idx[1])
        fig.add_trace(go.Histogram(x=df[f'{feature}']), **order)

    fig.update_layout(height=800, width=1000, showlegend=False,
                      title_text="Numerical features distribution")

    fig.show()
    return None


def corr_insights(df, label, color=None, opacity=0.05):
    corr_table = df.corr()[f'{label}'].sort_values(ascending=False, key=lambda x: abs(x)).to_frame()
    top_feat = corr_table.index[:4]
    label_clean = label.replace('_', ' ').title()
    values = [corr_table.index.str.replace('_', ' ').str.title()[1:],
              corr_table.iloc[1:, 0].round(3)]

    table = go.Figure(data=[go.Table(
        header=dict(
            values=['Features', f'Correlation (Pearson) with {label_clean}'],
            font_family='AkzidGrtskNext-Med',
            font_size=14,
            height=40,
            fill_color=water,
            align=['center', 'center'], ),
        cells=dict(
            values=values,
            font_family='AkzidGrtskNext-Regular',
            font_size=12,
            height=30,
            align=['center', 'center'],
            fill=dict(color=[bluewhite])
        ))
    ])

    table.update_layout(
        title='All Features',
        width=1000,
    )

    fig = px.scatter_matrix(df[top_feat],
                            color=color,
                            opacity=opacity,
                            labels={col: col.replace('_', ' ').title() for col in df.columns},
                            width=980, height=1000,
                            color_discrete_sequence=[water, pink, yellow, green, red]
                            )

    fig.update_traces(diagonal_visible=True)

    fig.update_layout(
        title='Most Correlated Features',
        dragmode='select',
        hovermode='closest')

    fig.show()
    table.show()
    return None


def map_the_label(df, latitude, longitude, label, size, divide_size_by=100, zoom=4):
    with open('/Users/alanperfettini/.tokens', 'r') as token:
        px.set_mapbox_access_token(token.read().strip())

    fig = px.scatter_mapbox(data_frame=df,
                            lon=longitude,
                            lat=latitude,
                            color=label,
                            size=df[f'{size}'] / divide_size_by,
                            zoom=zoom,
                            color_continuous_scale='Teal',
                            opacity=0.5, labels={label: label.replace('_', ' ').title()},
                            title='Geographical scope of label',
                            width=1000, height=600
                            )
    pio.show(fig)
    return None


def chart_na_values(df: pd.DataFrame, label) -> None:
    null_feat = pd.DataFrame(df[f'{label}'].size - df.isnull().sum(), columns=['Count'])
    null_feat.index = null_feat.index.str.replace('_', ' ').str.title()
    trace = go.Bar(x=null_feat.index,
                   y=null_feat['Count'],
                   opacity=0.8,
                   marker=dict(color='lightgrey',
                               line=dict(color='#000000',
                                         width=1.5)))

    layout = dict(title={'text': "Missing Values"})

    fig = dict(data=[trace], layout=layout)

    pio.show(fig)
    return None


def classification_model_performance_plot(model, y_test, y_pred, y_score):
    # conf matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    trace1 = go.Heatmap(z=conf_matrix, x=["0 (pred)", "1 (pred)"],
                        y=["0 (true)", "1 (true)"], xgap=2, ygap=2,
                        colorscale='GnBu', reversescale=True, showscale=False)

    # show metrics
    tp = conf_matrix[1, 1]
    fn = conf_matrix[1, 0]
    fp = conf_matrix[0, 1]
    tn = conf_matrix[0, 0]
    Accuracy = ((tp + tn) / (tp + tn + fp + fn))
    Precision = (tp / (tp + fp))
    Recall = (tp / (tp + fn))
    F1_score = (2 * (((tp / (tp + fp)) * (tp / (tp + fn))) / ((tp / (tp + fp)) + (tp / (tp + fn)))))

    show_metrics = pd.DataFrame(data=[[Accuracy, Precision, Recall, F1_score]])
    show_metrics = show_metrics.T

    colors = [yellow, green, pink, water]
    trace2 = go.Bar(x=show_metrics[0].values,
                    y=['Accuracy', 'Precision', 'Recall', 'F1'], text=np.round_(show_metrics[0].values, 4),
                    textposition='auto',
                    orientation='h', opacity=0.8, marker=dict(
            color=colors,
            line=dict(color='#000000', width=1.5)))

    # plot roc curve
    model_roc_auc = round(roc_auc_score(y_test, y_score), 3)
    fpr, tpr, t = roc_curve(y_test, y_score)
    trace3 = go.Scatter(x=fpr, y=tpr,
                        name="Roc : ",
                        line=dict(color=blue, width=2), fill='tozeroy')
    trace4 = go.Scatter(x=[0, 1], y=[0, 1],
                        line=dict(color='black', width=1.5,
                                  dash='dot'))

    # Precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_score)
    trace5 = go.Scatter(x=recall, y=precision,
                        name="Precision" + str(precision),
                        line=dict(color=pink, width=2), fill='tozeroy')

    # subplots
    fig = make_subplots(rows=2, cols=2, print_grid=False,
                        subplot_titles=('Confusion Matrix',
                                        'Metrics',
                                        'ROC curve' + " " + '(' + str(model_roc_auc) + ')',
                                        'Precision - Recall curve'))

    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 2)
    fig.append_trace(trace3, 2, 1)
    fig.append_trace(trace4, 2, 1)
    fig.append_trace(trace5, 2, 2)

    fig['layout'].update(showlegend=False, title='<b>Model performance</b><br>' + str(model),
                         autosize=False, height=800, width=1000,
                         # plot_bgcolor='rgba(240,240,240, 0.95)',
                         # paper_bgcolor='rgba(240,240,240, 0.95)',
                         margin=dict(b=195))
    fig["layout"]["xaxis2"].update((dict(range=[0, 1])))
    fig["layout"]["xaxis3"].update(dict(title="false positive rate"))
    fig["layout"]["yaxis3"].update(dict(title="true positive rate"))
    fig["layout"]["xaxis4"].update(dict(title="recall"), range=[0, 1.05])
    fig["layout"]["yaxis4"].update(dict(title="precision"), range=[0, 1.05])
    fig.layout.titlefont.size = 14

    py.iplot(fig)


# Theming graphs
pink, yellow, bluewhite, water, blue, green, red = ['rgb(247, 214, 214)', 'rgb(252, 245, 219)',
                                                    'rgb(247, 252, 255)', 'rgb(208, 234, 250)',
                                                    'rgb(197, 210, 239)', 'rgb(141, 227, 152)', 'rgb(227, 152, 141)']

pio.templates['custom_temp_blue'] = (pio.templates["plotly_white"]
                                     .update({'layout': dict(title_font_family="AkzidGrtskNext-Black",
                                                             title_font_size=22,
                                                             font_family='AkzidGrtskNext-Med',
                                                             yaxis={'gridcolor': 'rgb(255,255,255)'},
                                                             xaxis={'gridcolor': 'rgb(255,255,255)'},
                                                             font=dict(family="AkzidGrtskNext-Med", size=15),
                                                             paper_bgcolor='rgb(233,242,250)',
                                                             plot_bgcolor='rgb(233,242,250)',
                                                             uniformtext_minsize=8,
                                                             uniformtext_mode='hide')}))

pio.templates['custom_temp_grey'] = (pio.templates["plotly_white"]
                                     .update({'layout': dict(title_font_family="AkzidGrtskNext-Black",
                                                             title_font_size=22,
                                                             yaxis={'gridcolor': 'rgb(255,255,255)'},
                                                             xaxis={'gridcolor': 'rgb(255,255,255)'},
                                                             font_family='AkzidGrtskNext-Med',
                                                             font=dict(family="AkzidGrtskNext-Med", size=15),
                                                             paper_bgcolor='rgb(243,243,243)',
                                                             plot_bgcolor='rgb(243,243,243)',
                                                             uniformtext_minsize=8,
                                                             uniformtext_mode='hide')}))

pio.templates['custom_temp_white'] = (pio.templates["plotly_white"]
                                      .update({'layout': dict(title_font_family="AkzidGrtskNext-Black",
                                                              title_font_size=22,
                                                              font_family='AkzidGrtskNext-Med',
                                                              font=dict(family="AkzidGrtskNext-Med",
                                                                        size=15),
                                                              paper_bgcolor='rgb(255,255,255)',
                                                              plot_bgcolor='rgb(255,255,255)',
                                                              uniformtext_minsize=8,
                                                              uniformtext_mode='hide')}))

themes = cycle(['custom_temp_blue', 'custom_temp_grey', 'custom_temp_white'])


def change_theme(color):
    if color:
        pio.templates.default = f'custom_temp_{color}'
    else:
        pio.templates.default = next(themes)
    return None
