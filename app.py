from turtle import ht
import dash
from dash import dcc, html
import pandas as pd
import plotly.graph_objs as go

# Loading data
df = pd.read_csv("data/data.csv")  
raw_century = pd.read_csv("data/century_anomalies.csv")
rolling_year = pd.read_csv("data/rolling_mean_year.csv")
rolling_5 = pd.read_csv("data/rolling_mean_5years.csv")
rolling_10 = pd.read_csv("data/rolling_mean_10years.csv")

# Convert 'Date' column to datetime format
raw_century['Date'] = pd.to_datetime(raw_century['Date'])
rolling_year['Date'] = pd.to_datetime(rolling_year['Date'])
rolling_5['Date'] = pd.to_datetime(rolling_5['Date'])
rolling_10['Date'] = pd.to_datetime(rolling_10['Date'])

raw_century['Year'] = raw_century['Date'].dt.year
rolling_year['Year'] = rolling_year['Date'].dt.year
rolling_5['Year'] = rolling_5['Date'].dt.year
rolling_10['Year'] = rolling_10['Date'].dt.year

raw_fig = go.Scatter(x=raw_century['Date'], y=raw_century['Anomaly'], mode = 'lines', name='Global Temperature Anomalies (Raw Data)')
rolling_year_fig = go.Scatter(x=rolling_year['Date'], y=rolling_year['Anomaly'], mode='lines', name='Global Temperature Anomalies (Rolling Mean - 1 Year)')
rolling_5_fig = go.Scatter(x=rolling_5['Date'], y=rolling_5['Anomaly'], mode='lines', name='Global Temperature Anomalies (Rolling Mean - 5 Years)')
rolling_10_fig = go.Scatter(x=rolling_10['Date'], y=rolling_10['Anomaly'], mode='lines', name='Global Temperature Anomalies (Rolling Mean - 10 Years)')

forecast_data = pd.DataFrame({
    'Year': [2040, 2041, 2042, 2043, 2044, 2045],
    'Linear_prediction': [df[df['Year'] == year]['Linear_future'].values[0] for year in range(2040, 2046)],
    'Polynomial_prediction': [df[df['Year'] == year]['Poly_future'].values[0] for year in range(2040, 2046)],
    'Prophet_prediction': [df[df['Year'] == year]['Prophet_Trend'].values[0] for year in range(2040, 2046)]
})

forecast_data['Average'] = forecast_data[['Linear_prediction', 'Polynomial_prediction', 'Prophet_prediction']].mean(axis=1)

forecast_data['Linear_prediction'] = forecast_data['Linear_prediction'].round(2)
forecast_data['Polynomial_prediction'] = forecast_data['Polynomial_prediction'].round(2)
forecast_data['Prophet_prediction'] = forecast_data['Prophet_prediction'].round(2)
forecast_data['Average'] = forecast_data['Average'].round(2)

forecast_data.to_csv("data/forecast_data.csv", index=False)


app = dash.Dash(__name__)
app.title = "Global Temperature Anomaly Dashboard"

app.layout = html.Div([
    html.H1("Global Temperature Anomalies", style={'textAlign': 'center', 'color': '#fff', 'fontFamily': 'Arial'}),
    html.Div([
        html.P("This dashboard visualizes global temperature anomalies over the past 100 years, including Linear and Polynomial Regression lines as well as future predictions.", 
               style={'textAlign': 'center', 'color': '#fff', 'fontFamily': 'Arial'}),
    ], style={'marginBottom': '20px'}),

    html.H2("Data Visualizations and exploration", style={'textAlign': 'center', 'color': '#fff', 'fontFamily': 'Arial'}),
    html.Div([
        html.Div([
            dcc.Graph(
                id='raw-data',
                figure=go.Figure(data=[raw_fig], layout=go.Layout(title='Global Temperature Anomalies (Raw Data)', 
                                                                         font = dict(family='Arial', size=14, color='#fff'), 
                                                                         plot_bgcolor="#FCFEFF", paper_bgcolor="#2a2f31", 
                                                                         
                                                                         xaxis=dict(title='Year', showgrid=True, gridcolor='LightGray', 
                                                                                    gridwidth=1, zeroline=False),
                                                                         yaxis=dict(title='Anomaly (°C)', showgrid=True, gridcolor='LightGray',
                                                                                     gridwidth=1, zeroline=False)
                                                                    )   
                                                                    
                                                                ),
            )
        ], style={'width': '48%', 'display': 'inline-block'}),

        html.Div([
            dcc.Graph(
                id='rolling-year',
                figure=go.Figure(data=[rolling_year_fig], layout=go.Layout(title='Global Temperature Anomalies (Rolling Mean - 1 Year)', 
                                                                         font = dict(family='Arial', size=14, color='#fff'), 
                                                                         plot_bgcolor="#FCFEFF", paper_bgcolor="#2a2f31",
                                                                         xaxis=dict(title='Year', showgrid=True, gridcolor='LightGray', 
                                                                                    gridwidth=1, zeroline=False),
                                                                         yaxis=dict(title='Anomaly (°C)', showgrid=True, gridcolor='LightGray',
                                                                                     gridwidth=1, zeroline=False))),
            )
        ], style={'width': '48%', 'display': 'inline-block'})
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '5px'}),

    html.Div([
        html.Div([
            dcc.Graph(
                id='rolling-5',
                figure=go.Figure(data=[rolling_5_fig], layout=go.Layout(title='Global Temperature Anomalies (Rolling Mean - 5 Years)', 
                                                                         font = dict(family='Arial', size=14, color='#fff'), 
                                                                         plot_bgcolor="#FCFEFF", paper_bgcolor="#2a2f31",
                                                                         xaxis=dict(title='Year', showgrid=True, gridcolor='LightGray', 
                                                                                    gridwidth=1, zeroline=False),
                                                                         yaxis=dict(title='Anomaly (°C)', showgrid=True, gridcolor='LightGray',
                                                                                     gridwidth=1, zeroline=False))),
            )
        ], style={'width': '48%', 'display': 'inline-block'}),

        html.Div([
            dcc.Graph(
                id='rolling-10',
                figure=go.Figure(data=[rolling_10_fig], layout=go.Layout(title='Global Temperature Anomalies (Rolling Mean - 10 Years)', 
                                                                         font = dict(family='Arial', size=14, color='#fff'), 
                                                                         plot_bgcolor="#FCFEFF", paper_bgcolor="#2a2f31", 
                                                                         
                                                                         xaxis=dict(title='Year', showgrid=True, gridcolor='LightGray', 
                                                                                    gridwidth=1, zeroline=False),
                                                                         yaxis=dict(title='Anomaly (°C)', showgrid=True, gridcolor='LightGray',
                                                                                     gridwidth=1, zeroline=False))),
            )
        ], style={'width': '48%', 'display': 'inline-block'})
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '5px'}),

    
    html.H2("Temperature Anomalies with Trends and Predictions", style={'textAlign': 'center', 'color': '#fff', 'fontFamily': 'Arial'}),
    html.Div([
        html.P("This section includes Linear and Polynomial Regression lines on the yearly data, as well as future prediction lines for global temperature anomalies up to the year 2045.", 
               style={'textAlign': 'center', 'color': '#fff', 'fontFamily': 'Arial'}),
        html.P("The data used for the models was the mean yearly data from the last 100 years.",
               style={'textAlign': 'center', 'color': '#fff', 'fontFamily': 'Arial'}),
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
            font=dict(family='Arial', size=14, color='#fff'),
            paper_bgcolor="#2a2f31",
            plot_bgcolor="#FCFEFF",
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

        ),
        html.Div([
            html.H3("Table of future predicitons from all the models used in this dashboard (2040-2045)", style={'textAlign': 'center', 'color': '#fff', 'fontFamily': 'Arial'}),
        ]),

            dcc.Graph(
                id='forecast-table',
                figure={
                    'data': [
                        go.Table(
                            header=dict(values=['Year', 'Linear Prediction', 'Polynomial Prediction', 'Prophet Prediction', 'Average'],
                                        fill_color='paleturquoise',
                                        align='left'),
                            cells=dict(values=[forecast_data['Year'], forecast_data['Linear_prediction'], 
                                               forecast_data['Polynomial_prediction'], forecast_data['Prophet_prediction'], forecast_data['Average']],
                                       fill_color='lavender',
                                       align='left'))
                    ],
                    'layout': go.Layout(title=dict(text='Future Predictions (2040-2045)', font = dict(family='Arial', size=16, color='#fff')),
                                       font=dict(family='Arial', size=14, color='#000'),
                                       paper_bgcolor="#2a2f31", plot_bgcolor="#FCFEFF")
                }
            ),
           html.H3("Data", style={'textAlign': 'center', 'color': '#fff', 'fontFamily': 'Arial'}),
            html.P("The data, libraries and models used in this dashboard are the following:",
                   style={'textAlign': 'center', 'color': '#fff', 'fontFamily': 'Arial'}),
            html.Ul([
                html.Li("Anomalies: NOAA National Centers for Environmental Information (NCEI)"),
                html.Li("Rolling Means: Calculated from the raw data using Pandas"),
                html.Li("Forecasting Models: Linear Regression, Polynomial Regression, and Prophet"),
                html.Li("Data Visualization: Plotly and Dash")
            ], style={'textAlign': 'center', 'color': '#fff', 'fontFamily': 'Arial'}),
            html.P("created by: Muhammed Panjwani",
                   style={'textAlign': 'center', 'color': '#fff', 'fontFamily': 'Arial'}),
            

], style={
    'backgroundColor': "#2a2f31",
    'minHeight': '100vh',          
    'padding': '20px',             
    'fontFamily': 'Arial'          
})

if __name__ == '__main__':
    app.run(debug=True)
