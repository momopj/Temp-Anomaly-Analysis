from pydoc import doc
import dash
from dash import dcc, html
import pandas as pd
import plotly.graph_objs as go

# Load your data
df = pd.read_csv("data.csv")  # should include Year, Anomaly, Poly_Pred, Linear_Pred, etc.

app = dash.Dash(__name__)
app.title = "Global Temperature Anomaly Dashboard"

app.layout = html.Div([
    html.H1("Global Temperature Anomalies", style={'textAlign': 'center', 'color': '#333', 'fontFamily': 'Arial'}),
    html.Div([
        html.P("This dashboard visualizes global temperature anomalies over the years, including linear and polynomial trend lines as well as future predictions.", 
               style={'textAlign': 'center', 'color': '#666', 'fontFamily': 'Arial'}),
    ], style={'marginBottom': '20px'}),
    

    dcc.Graph(
    id='anomaly-graph',
    figure={
        'data': [
            go.Scatter(x=df['Year'], y=df['Anomaly'], mode='markers', name='Observed'),
            go.Scatter(x=df['Year'], y=df['Predicted_linear'], mode='lines', name='Linear Trend'),
            go.Scatter(x=df['Year'], y=df['Predicted_poly'], mode='lines', name='Polynomial Trend'),
            go.Scatter(x=df['Year'], y=df['Linear_future'], mode='lines', name='Future Linear Prediction'),
            go.Scatter(x=df['Year'], y=df['Poly_future'], mode='lines', name='Future Polynomial Prediction'),
            go.Scatter(x=df['Year'], y=df['Prophet_Trend'], mode='lines', name='Prophet Trend'),
        ],
        'layout': go.Layout(
            title=dict(text = 'Temperature Anomalies with Trends and Predictions (to 2045)'),
            xaxis={'title': 'Year'},
            yaxis={'title': 'Anomaly (°C)'},
            hovermode='closest',
            updatemenus=[
                {
                    'buttons': [
                        {'label': 'Linear Regression', 'method': 'update',
                         'args': [{'visible': [True, True, False, False, False, False]},
                                  {'title':{'text' : 'Linear Regression on Global Temperature Anomalies'}}]},
                        {'label': 'Polynomial Regression', 'method': 'update',
                         'args': [{'visible': [True, False, True, False, False, False]},
                                  {'title': {'text': 'Polynomial Regression on Global Temperature Anomalies'}}]},
                        {'label': 'Linear Prediction', 'method': 'update',
                         'args': [{'visible': [True, False, False, True, False, False]},
                                  {'title': {'text': 'Linear Prediction for Global Temperature Anomalies (to 2045)'}}]},
                        {'label': 'Polynomial Prediction', 'method': 'update',
                         'args': [{'visible': [True, False, False, False, True, False]},
                                  {'title': {'text': 'Polynomial Prediction for Global Temperature Anomalies (to 2045)'}}]},
                        {'label': 'Prophet Trend', 'method': 'update',
                         'args': [{'visible': [True, False, False, False, False, True]},
                                  {'title': {'text': 'Prophet Trend on Global Temperature Anomalies (to 2045)'}}]},
                        {'label': 'Linear + Polynomial Regressions', 'method': 'update',
                         'args': [{'visible': [True, True, True, False, False, False]},
                                  {'title': {'text': 'Linear and Polynomial Regression on Global Temperature Anomalies'}}]},
                        {'label': 'Linear Regression + Prediction', 'method': 'update',
                         'args': [{'visible': [True, True, False, True, False, False]},
                                  {'title': {'text': 'Linear Regression + Predictions on Global Temperature Anomalies (to 2045)'}}]},
                        {'label': 'Polynomial Regression + Prediction', 'method': 'update',
                         'args': [{'visible': [True, False, True, False, True, False]},
                                  {'title': {'text': 'Polynomial Regression + Predictions on Global Temperature Anomalies (to 2045)'}}]},
                        {'label': 'All Regressions', 'method': 'update',
                         'args': [{'visible': [True, True, True, True, True, False]},
                                  {'title': {'text': 'Linear + Polynomial Regressions with Predictions on Global Temperature Anomalies (to 2045)'}}]},
                        {'label': 'Show All', 'method': 'update',
                         'args': [{'visible': [True, True, True, True, True, True]},
                                  {'title': {'text': 'All Models and Predictions on Global Temperature Anomalies (to 2045)'}}]},
                    ],
                    'direction': 'down',
                    'showactive': True,
                    'x': 1.2,
                    'xanchor': 'left',
                    'y': 1.1,
                    'yanchor': 'top'
                }
            ]
        )
    }
)

])

if __name__ == '__main__':
    app.run(debug=True)
