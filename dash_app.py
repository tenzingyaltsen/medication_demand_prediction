import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta
from tensorflow import keras
import joblib

# Load model, scaler, and input structure
nn = keras.models.load_model("final_ffnn_model.keras")
scaler = joblib.load("scaler.joblib")
input_columns = joblib.load("input_columns.joblib")

# Load and process dataset
df = pd.read_csv("medication_demand_data.csv") 
df['Date'] = pd.to_datetime(df['Date'])

# EDA: Data copy and transformation
df_eda = df.copy()
df_eda['GT_rounded'] = df_eda['Google_Trends'].round()
df_eda['Spend_bin'] = pd.qcut(df_eda['Marketing_Spend'], q=10, duplicates='drop')

# SARIMA preparation
df_grouped = df.groupby('Date')['Sales'].sum().asfreq('D').ffill()
sarima_model = SARIMAX(df_grouped, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
sarima_results = sarima_model.fit(disp=False)

# App setup
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "ML Dashboard"

# ---------------- Tabs ----------------

def layout_background_eda():
    # Plot 1: Google Trends vs Sales
    agg_line = df_eda.groupby(['GT_rounded', 'Medication'], observed=True).agg(Avg_Sales=('Sales', 'mean')).reset_index()
    fig1 = px.scatter(agg_line, x='GT_rounded', y='Avg_Sales', color='Medication',
                      trendline='ols', title='Average Sales vs Google Trends (with Fitted Trendlines)',
                      labels={'GT_rounded': 'Google Trends', 'Avg_Sales': 'Average Sales'})
    fig1.update_layout(title_x=0.5)

    # Plot 2: Spend vs Sales
    agg_bin = df_eda.groupby(['Spend_bin', 'Medication'], observed=True).agg(
        Avg_Sales=('Sales', 'mean'), Avg_Spend=('Marketing_Spend', 'mean')).reset_index()
    agg_bin['Spend_bin'] = agg_bin['Spend_bin'].astype(str)
    fig2 = px.line(agg_bin, x='Avg_Spend', y='Avg_Sales', color='Medication', markers=True,
                   title='Average Sales by Binned Marketing Spend and Medication')
    fig2.update_layout(title_x=0.5)

    # Plot 3: Heatmap by Season
    df['Month'] = df['Date'].dt.month_name()
    season_map = {
        'January': 'Winter', 'February': 'Winter', 'March': 'Spring', 'April': 'Spring',
        'May': 'Spring', 'June': 'Summer', 'July': 'Summer', 'August': 'Summer',
        'September': 'Fall', 'October': 'Fall', 'November': 'Fall', 'December': 'Winter'
    }
    df['Season'] = df['Month'].map(season_map)
    med_month = df.groupby(['Medication', 'Season'], observed=True)['Sales'].mean().reset_index()
    heatmap_data = med_month.pivot(index='Medication', columns='Season', values='Sales')
    heatmap_data = heatmap_data[['Winter', 'Spring', 'Summer', 'Fall']]
    fig3 = px.imshow(heatmap_data, text_auto='.1f', color_continuous_scale='YlOrRd',
                     labels=dict(x='Season', y='Medication', color='Avg Sales'),
                     title='Medication Sales by Season (Heatmap)')
    fig3.update_layout(title_x=0.5, width=900, height=500)

    # Plot 4: Region Trends
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    region_over_time = df.groupby(['Month', 'Region'], observed=True)['Sales'].mean().reset_index()
    region_over_time['Month'] = pd.Categorical(region_over_time['Month'], categories=month_order, ordered=True)
    region_over_time = region_over_time.sort_values('Month')
    fig4 = px.line(region_over_time, x='Month', y='Sales', color='Region', markers=True,
                   title='Sales Trends by Region (Line Plot)')
    fig4.update_layout(title_x=0.5)

    # Plot 5: Allergy vs Pollen
    allergy_df = df_eda[df_eda['Medication'] == 'Allergy Relief']
    fig5 = px.scatter(allergy_df, x='Pollen_Count', y='Sales', trendline='ols',
                      title='Allergy Relief Sales vs Pollen Count', opacity=0.6)
    fig5.update_layout(title_x=0.5)

    return html.Div([
        html.H2("Background and Exploratory Data Analysis"),
        dcc.Graph(figure=fig1),
        dcc.Graph(figure=fig2),
        dcc.Graph(figure=fig3),
        dcc.Graph(figure=fig4),
        dcc.Graph(figure=fig5),
    ])

def layout_model_metrics():
    return html.Div([
        html.H2("Model Performance Metrics"),
        html.Label("Select a model:"),
        dcc.Dropdown(
            id='model-selector',
            options=[{'label': m, 'value': m} for m in ['XGB', 'RandomForest', 'DT', 'KNN', 'Linear', 'AdaBoost', 'FFNN', 'SARIMA']],
            value='XGB'
        ),
        html.Br(),
        html.Div(id='model-metrics-output')
    ])

def layout_nn_predictions():
    return html.Div([
        html.H2("Neural Network Sales Prediction"),
        html.Label("Google Trends Score:"), dcc.Input(id='input-google-trends', type='number', value=50),
        html.Label("Marketing Spend:"), dcc.Input(id='input-marketing', type='number', value=2000),
        html.Label("Pollen Count:"), dcc.Input(id='input-pollen', type='number', value=30),
        html.Label("Temperature (°C):"), dcc.Slider(id='input-temp', min=-10, max=40, step=1, value=20),
        html.Label("Region:"), dcc.Dropdown(id='input-region',
            options=[{'label': r, 'value': r} for r in ['North', 'South', 'East', 'West']], value='North'),
        html.Label("Medication:"), dcc.Dropdown(id='input-medication',
            options=[{'label': m, 'value': m} for m in ['Pain Relief', 'Allergy Relief', 'Cold & Flu']], value='Pain Relief'),
        html.Br(), html.Button("Predict", id='predict-button'),
        html.H4("Predicted Sales:"), html.Div(id='nn-prediction-output')
    ])

def layout_time_series():
    return html.Div([
        html.H2("Time Series Forecast (SARIMA)"),
        html.Label("Enter number of days to forecast:"),
        dcc.Input(id='forecast-days', type='number', value=30, min=1, max=365),
        html.Br(), html.Button("Generate Forecast", id='forecast-button'),
        html.Br(), dcc.Graph(id='forecast-plot')
    ])

# ---------------- Layout ----------------

app.layout = html.Div([
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Background & EDA', value='tab-1'),
        dcc.Tab(label='Model Metrics', value='tab-2'),
        dcc.Tab(label='Neural Network Predictions', value='tab-3'),
        dcc.Tab(label='Time Series Forecast', value='tab-4'),
    ]),
    html.Div(id='tabs-content')
])

@app.callback(Output('tabs-content', 'children'), Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return layout_background_eda()
    elif tab == 'tab-2':
        return layout_model_metrics()
    elif tab == 'tab-3':
        return layout_nn_predictions()
    elif tab == 'tab-4':
        return layout_time_series()

# ---------------- Callbacks ----------------

@app.callback(
    Output('model-metrics-output', 'children'),
    Input('model-selector', 'value')
)
def update_model_metrics(model_name):
    metrics = {
        'XGB': {'R² Score': 0.777, 'RMSE': 215.56, 'MSE': 46464.21},
        'RandomForest': {'R² Score': 0.772, 'RMSE': 217.95, 'MSE': 47501.41},
        'DT': {'R² Score': 0.740, 'RMSE': 232.93, 'MSE': 54257.89},
        'KNN': {'R² Score': 0.554, 'RMSE': 304.78, 'MSE': 92890.29},
        'Linear': {'R² Score': 0.536, 'RMSE': 310.78, 'MSE': 96581.46},
        'AdaBoost': {'R² Score': -0.052, 'RMSE': 468.13, 'MSE': 219146.17},
        'FFNN': {'R² Score': 'N/A', 'RMSE': 198.95, 'MSE': 39579.45},
        'SARIMA': {'R² Score': 'N/A', 'RMSE': 5204.79, 'MSE': 2709807.14}
    }
    selected = metrics[model_name]
    return html.Div([
        html.H4(f"Metrics for {model_name}:"),
        html.Ul([
            html.Li(f"R² Score: {selected['R² Score']}"),
            html.Li(f"RMSE: {selected['RMSE']}"),
            html.Li(f"MSE: {selected['MSE']}")
        ])
    ])

@app.callback(
    Output('nn-prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    State('input-google-trends', 'value'),
    State('input-marketing', 'value'),
    State('input-pollen', 'value'),
    State('input-temp', 'value'),
    State('input-region', 'value'),
    State('input-medication', 'value')
)
def predict_sales(n_clicks, google_trends, marketing, pollen, temp, region, medication):
    if not n_clicks:
        return ""
    df_input = pd.DataFrame([{
        'Google_Trends': google_trends,
        'Marketing_Spend': marketing,
        'Pollen_Count': pollen,
        'Temperature': temp,
        'Region': region,
        'Medication': medication
    }])
    df_input = pd.get_dummies(df_input)
    for col in input_columns:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[input_columns]
    scaled_input = scaler.transform(df_input)
    prediction = nn.predict(scaled_input)[0][0]
    return f"{prediction:.2f} units"

@app.callback(
    Output('forecast-plot', 'figure'),
    Input('forecast-button', 'n_clicks'),
    State('forecast-days', 'value')
)
def update_forecast(n_clicks, days):
    if not n_clicks or not days:
        return go.Figure()
    forecast_result = sarima_results.get_forecast(steps=days)
    forecast_index = pd.date_range(start=df_grouped.index[-1] + timedelta(days=1), periods=days)
    forecast_series = pd.Series(forecast_result.predicted_mean.values, index=forecast_index)
    conf_int = forecast_result.conf_int()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_grouped.index, y=df_grouped.values, mode='lines', name='Historical'))
    fig.add_trace(go.Scatter(x=forecast_index, y=forecast_series, mode='lines', name='Forecast'))
    fig.add_trace(go.Scatter(
        x=forecast_index.tolist() + forecast_index[::-1].tolist(),
        y=conf_int.iloc[:, 0].tolist() + conf_int.iloc[:, 1][::-1].tolist(),
        fill='toself',
        fillcolor='rgba(255,165,0,0.3)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        name='Confidence Interval'
    ))
    fig.update_layout(title=f"Sales Forecast for Next {days} Days",
                      xaxis_title="Date", yaxis_title="Sales", template="plotly_white")
    return fig

# ---------------- Run ----------------

if __name__ == '__main__':
    app.run(debug=True)
