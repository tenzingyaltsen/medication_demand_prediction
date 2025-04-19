import dash
"""
Medication Demand Forecasting Dashboard (Dash App)
--------------------------------------------------
This app predicts medication sales using historical data,
time series forecasting (SARIMA), and a trained feedforward neural network (FFNN).
It provides visual EDA, model performance metrics, and both manual and automatic forecasts.

Author: [Your Name or Team]
Version: 1.0
"""
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta
from tensorflow import keras
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load model, scaler, and column order
nn = keras.models.load_model("final_ffnn_model.keras")
scaler = joblib.load("scaler.joblib")
input_columns = joblib.load("input_columns.joblib")

# Load data
df = pd.read_csv("medication_demand_data.csv")
df['Date'] = pd.to_datetime(df['Date'])
df_clean = pd.read_csv("clean_medication_data.csv")
df_clean['Date'] = pd.to_datetime(df_clean['Date'])  

df_clean['Month'] = df_clean['Date'].dt.month_name()
season_map = {
    'January': 'Winter', 'February': 'Winter', 'March': 'Spring', 'April': 'Spring',
    'May': 'Spring', 'June': 'Summer', 'July': 'Summer', 'August': 'Summer',
    'September': 'Fall', 'October': 'Fall', 'November': 'Fall', 'December': 'Winter'
}
df_clean['Season'] = df_clean['Month'].map(season_map)
# Compute default values for inputs
def_mode = lambda col: sorted(df_clean[col].mode())[0] if not df_clean[col].mode().empty else None

default_vals = {
    'Temperature': round(df_clean['Temperature'].mean(), 1),
    'Humidity': round(df_clean['Humidity'].mean(), 1),
    'Flu_Cases': round(df_clean['Flu_Cases'].mean(), 1),
    'Pollen_Count': round(df_clean['Pollen_Count'].mean(), 1),
    'Google_Trends': round(df_clean['Google_Trends'].mean(), 1),
    'Marketing_Spend': round(df_clean['Marketing_Spend'].mean(), 1),
    'Medication': def_mode('Medication'),
    'Region': def_mode('Region'),
    'Holiday': def_mode('Holiday'),
    'Season': def_mode('Season')
}

# SARIMA training data
df_grouped = df.groupby('Date')['Sales'].sum().asfreq('D').ffill()
sarima_model = SARIMAX(df_grouped, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
sarima_results = sarima_model.fit(disp=False)

# Initialize app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Medication Demand Prediction"

# REQUIRED for Render Deployment
server = app.server


# ----------------- Layouts -----------------

# Layout for EDA Tab with Dropdown
@app.callback(
    Output('eda-plot-container', 'children'),
    Input('eda-plot-selector', 'value')
)
def update_eda_plot(selected_plot):
    df_eda = df.copy()

    numeric_vars = ['Sales', 'Temperature', 'Humidity', 'Flu_Cases', 'Pollen_Count', 'Google_Trends', 'Marketing_Spend']
    categorical_vars = ['Medication', 'Region', 'Holiday']

    blurbs = {
        'Histograms of Numeric Variables': html.P([
            "We can see that our outcome variable, Sales, is pretty right skewed. Next is Temperature, where we can see almost a bimodal distribution with 2 peaks. With respect to Humidity, we see a nicely normal distribution centred around 67% relative humidity. We can see that Flu Cases, Pollen Count and Marketing Spend display varying degrees of right skewness and also contain very large outliers. Conversely, the Google Trends variable is much more normally distributed, centred around a mean score of about 63.", html.Br(),
            html.Strong("NOTE: "),
            "The purpose of these high-level \"distribution\" or \"count\"-based EDA plots is to provide a quick visual understanding of the variables' structure and spread. These plots are not intended to generate business insights at this stage, but rather to inform and guide future preprocessing and analytical decisions."
        ]),
        'Bar Charts of Categorical Variables': html.P([
            "The majority of classes are equally distributed, including the different medications, the different regions, the different months and the different seasons in the data set. The only exception is the ‘Holiday’ variable where most of observations belonged to the ‘Missing’ or ‘Not a Holiday’ class.", html.Br(),
            html.Strong("NOTE: "),
            "The purpose of these high-level \"distribution\" or \"count\"-based EDA plots is to provide a quick visual understanding of the variables' structure and spread. These plots are not intended to generate business insights at this stage, but rather to inform and guide future preprocessing and analytical decisions."
        ]),
        'Average Sales vs Google Trends (with Fitted Trendlines)': "This plot supports the idea that Google Trends can serve as a leading indicator of medication demand, as all fitted trendlines show a positive slope. This suggests that higher search interest is generally associated with increased sales.\n\nMedications tied to seasonal or outbreak-driven symptoms, such as cough suppressants, cold relief, and allergy medications which exhibit the steepest slopes, indicating that their sales may be more sensitive to changes in public interest.\n\nA higher slope reflects a stronger rate of increase in sales with rising search volume, making these categories potentially more predictable or responsive in demand forecasting efforts.",
        'Average Sales by Binned Marketing Spend and Medication': "This plot illustrates how average sales change across increasing levels of marketing spend, grouped into ten bins to smooth out noise and highlight trends.\n\nCold Relief and Allergy Relief stand out as the most responsive to marketing investment. Their average sales rise significantly with higher spending, particularly in the upper bins. Cold Relief shows a strong lift early on but begins to level off, while Allergy Relief continues to climb more steadily across the full range.\n\nIn contrast, Pain Relief and Fever Reducer show more moderate responsiveness. Their sales increase slightly with higher spend but tend to plateau, indicating a point of diminishing returns.\n\nCough Suppressant displays a relatively flat trend throughout and even appears to decline at higher spending levels. This suggests that beyond a certain threshold, additional marketing may not translate into increased sales for that category, possibly due to saturation or consumer limits.\n\nOverall, this chart reveals that not all medications respond equally to marketing investment. Strategic spending on products like Cold Relief and Allergy Relief could offer better returns, while others may require a more cautious approach to avoid overspending where it has limited effect.",
        'Medication Sales by Season (Heatmap)': "This heatmap shows the average sales of each medication across the four seasons, revealing distinct seasonal patterns in demand.\n\nAllergy Relief peaks sharply in the Spring, which aligns with the start of allergy season. Its sales are significantly lower in all other seasons, especially Winter and Fall.\n\nCold Relief and Cough Suppressant show the opposite trend, with their highest sales occurring in Winter, tapering off as the year progresses. This is consistent with cold and flu season occurring during colder months.\n\nFever Reducer sees relatively high sales in Winter but maintains moderate demand throughout the year, suggesting more consistent usage or broader symptom application.\n\nPain Relief shows the most stable sales across seasons, with very little fluctuation. This likely indicates year-round demand unrelated to seasonality.\nThis seasonal breakdown helps identify when different medications are most in demand. It suggests that marketing and supply strategies (for businesses) should be seasonally adjusted.",
        'Sales Trends by Region (Line Plot)': "This plot shows that average sales peak during winter months (January, February, and December) across all regions, likely due to seasonal factors like cold and flu activity.\n\nSales begin to decline through March to May, but the lowest levels are reached in the summer, especially in June, July, and August.\n\nAfter that, sales gradually rise again through fall, with a sharp increase in December.\n\nToronto shows the highest average sales overall, while Vancouver tends to trend slightly lower. Despite differences in magnitude, all regions seem to follow a similar seasonal sales pattern.",
        'Allergy Relief Sales vs Pollen Count': "This scatter plot shows a positive relationship between pollen count and Allergy Relief sales. As pollen levels rise, sales generally increase as well. The upward trendline indicates that higher pollen exposure could be associated with greater demand for Allergy Relief products. Despite some variation at extreme values, the overall pattern suggests a seasonal effect where allergy-related sales respond to environmental triggers (pollen)."
    }

    def wrap_plot(fig, title):
        return html.Div([
            html.P(blurbs[title], style={"marginBottom": "10px", "textAlign": "left"}),
            dcc.Graph(figure=fig)
        ])

    if selected_plot == 'Histograms of Numeric Variables':
        cols_num = 2
        rows_num = -(-len(numeric_vars) // cols_num)
        fig = make_subplots(rows=rows_num, cols=cols_num, subplot_titles=numeric_vars)
        for i, col in enumerate(numeric_vars):
            row = i // cols_num + 1
            col_pos = i % cols_num + 1
            fig.add_trace(go.Histogram(x=df_eda[col], name=col), row=row, col=col_pos)
        fig.update_layout(title_text=selected_plot, height=300 * rows_num, showlegend=False)
        return wrap_plot(fig, selected_plot)

    if selected_plot == 'Bar Charts of Categorical Variables':
        cols_cat = 2
        rows_cat = -(-len(categorical_vars) // cols_cat)
        fig = make_subplots(rows=rows_cat, cols=cols_cat, subplot_titles=categorical_vars)
        for i, col in enumerate(categorical_vars):
            row = i // cols_cat + 1
            col_pos = i % cols_cat + 1
            filled = df_eda[col].fillna('Missing').astype(str)
            counts = filled.value_counts().reset_index()
            counts.columns = ['Category', 'Count']
            fig.add_trace(go.Bar(x=counts['Category'], y=counts['Count'], name=col), row=row, col=col_pos)
            fig.update_xaxes(tickangle=45, row=row, col=col_pos)
        fig.update_layout(title_text=selected_plot, height=300 * rows_cat, showlegend=False)
        return wrap_plot(fig, selected_plot)

    df_eda['GT_rounded'] = df_eda['Google_Trends'].round()
    if selected_plot == 'Average Sales vs Google Trends (with Fitted Trendlines)':
        agg_line = df_eda.groupby(['GT_rounded', 'Medication'], observed=True).agg(Avg_Sales=('Sales', 'mean')).reset_index()
        fig = px.scatter(agg_line, x='GT_rounded', y='Avg_Sales', color='Medication', trendline='ols', title=selected_plot)
        return wrap_plot(fig, selected_plot)

    if selected_plot == 'Average Sales by Binned Marketing Spend and Medication':
        df_eda['Spend_bin'] = pd.qcut(df_eda['Marketing_Spend'], q=10, duplicates='drop')
        agg_bin = df_eda.groupby(['Spend_bin', 'Medication'], observed=True).agg(
            Avg_Sales=('Sales', 'mean'), Avg_Spend=('Marketing_Spend', 'mean')).reset_index()
        agg_bin['Spend_bin'] = agg_bin['Spend_bin'].astype(str)
        fig = px.line(agg_bin, x='Avg_Spend', y='Avg_Sales', color='Medication', markers=True, title=selected_plot)
        return wrap_plot(fig, selected_plot)

    if selected_plot == 'Medication Sales by Season (Heatmap)':
        df_eda['Month'] = df_eda['Date'].dt.month_name()
        season_map = {'January': 'Winter', 'February': 'Winter', 'March': 'Spring', 'April': 'Spring',
                      'May': 'Spring', 'June': 'Summer', 'July': 'Summer', 'August': 'Summer',
                      'September': 'Fall', 'October': 'Fall', 'November': 'Fall', 'December': 'Winter'}
        df_eda['Season'] = df_eda['Month'].map(season_map)
        med_month = df_eda.groupby(['Medication', 'Season'], observed=True)['Sales'].mean().reset_index()
        heatmap_data = med_month.pivot(index='Medication', columns='Season', values='Sales')[['Winter', 'Spring', 'Summer', 'Fall']]
        fig = px.imshow(heatmap_data, text_auto='.1f', color_continuous_scale='YlOrRd', title=selected_plot)
        return wrap_plot(fig, selected_plot)

    if selected_plot == 'Sales Trends by Region (Line Plot)':
        df_eda['Month'] = df_eda['Date'].dt.month_name()
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']
        df_eda['Month'] = pd.Categorical(df_eda['Month'], categories=month_order, ordered=True)
        region_over_time = df_eda.groupby(['Month', 'Region'], observed=True)['Sales'].mean().reset_index()
        region_over_time = region_over_time.sort_values('Month')
        fig = px.line(region_over_time, x='Month', y='Sales', color='Region', markers=True, title=selected_plot)
        return wrap_plot(fig, selected_plot)

    if selected_plot == 'Allergy Relief Sales vs Pollen Count':
        allergy_df = df_eda[df_eda['Medication'] == 'Allergy Relief']
        fig = px.scatter(allergy_df, x='Pollen_Count', y='Sales', trendline='ols', title=selected_plot)
        return wrap_plot(fig, selected_plot)

    return html.Div("")

def layout_background_eda():
    return html.Div([
        html.H2("Background and Exploratory Data Analysis"),
        html.P(
            "Accurately predicting medication demand is crucial for ensuring a steady supply of essential drugs and preventing shortages. "
            "Various factors, such as flu outbreaks, allergy seasons, and trends, influence demand across regions. "
            "By analyzing historical data and external factors, and utilizing multiple modelling approaches — including both statistical forecasting (SARIMA), "
            "machine learning models (like Linear Regression and Random Forest), and neural networks — we can better capture different aspects of demand patterns.",
            style={"marginBottom": "40px", "lineHeight": "1.6", "textAlign": "left"}
        ),
        html.Label("Select an EDA Plot to View:"),
        dcc.Dropdown(
            id='eda-plot-selector',
            options=[
                {'label': 'Histograms of Numeric Variables', 'value': 'Histograms of Numeric Variables'},
                {'label': 'Bar Charts of Categorical Variables', 'value': 'Bar Charts of Categorical Variables'},
                {'label': 'Average Sales vs Google Trends (with Fitted Trendlines)', 'value': 'Average Sales vs Google Trends (with Fitted Trendlines)'},
                {'label': 'Average Sales by Binned Marketing Spend and Medication', 'value': 'Average Sales by Binned Marketing Spend and Medication'},
                {'label': 'Medication Sales by Season (Heatmap)', 'value': 'Medication Sales by Season (Heatmap)'},
                {'label': 'Sales Trends by Region (Line Plot)', 'value': 'Sales Trends by Region (Line Plot)'},
                {'label': 'Allergy Relief Sales vs Pollen Count', 'value': 'Allergy Relief Sales vs Pollen Count'}
            ],
            placeholder='Select a visualization...'
        ),
        html.Div(id='eda-plot-container', style={"marginTop": "20px"})
    ])

    # --- Categorical Distribution Plot ---
    categorical_vars = ['Medication', 'Region', 'Holiday']
    cols_cat = 2
    rows_cat = -(-len(categorical_vars) // cols_cat)
    fig_categorical = make_subplots(rows=rows_cat, cols=cols_cat, subplot_titles=categorical_vars)
    for i, col in enumerate(categorical_vars):
        row = i // cols_cat + 1
        col_pos = i % cols_cat + 1
        filled = df_eda[col].fillna('Missing').astype(str)
        counts = filled.value_counts().reset_index()
        counts.columns = ['Category', 'Count']
        fig_categorical.add_trace(go.Bar(x=counts['Category'], y=counts['Count'], name=col), row=row, col=col_pos)
        fig_categorical.update_xaxes(tickangle=45, row=row, col=col_pos)
    fig_categorical.update_layout(title_text='Bar Charts of Categorical Variables', height=300 * rows_cat, showlegend=False)

    df_eda['GT_rounded'] = df_eda['Google_Trends'].round()
    agg_line = df_eda.groupby(['GT_rounded', 'Medication'], observed=True).agg(Avg_Sales=('Sales', 'mean')).reset_index()
    fig1 = px.scatter(agg_line, x='GT_rounded', y='Avg_Sales', color='Medication', trendline='ols',
                      title='Average Sales vs Google Trends (with Fitted Trendlines)')
    df_eda['Spend_bin'] = pd.qcut(df_eda['Marketing_Spend'], q=10, duplicates='drop')
    agg_bin = df_eda.groupby(['Spend_bin', 'Medication'], observed=True).agg(
        Avg_Sales=('Sales', 'mean'), Avg_Spend=('Marketing_Spend', 'mean')).reset_index()
    agg_bin['Spend_bin'] = agg_bin['Spend_bin'].astype(str)
    fig2 = px.line(agg_bin, x='Avg_Spend', y='Avg_Sales', color='Medication', markers=True,
                   title='Average Sales by Binned Marketing Spend and Medication')
    df['Month'] = df['Date'].dt.month_name()
    season_map = {'January': 'Winter', 'February': 'Winter', 'March': 'Spring', 'April': 'Spring',
                  'May': 'Spring', 'June': 'Summer', 'July': 'Summer', 'August': 'Summer',
                  'September': 'Fall', 'October': 'Fall', 'November': 'Fall', 'December': 'Winter'}
    df['Season'] = df['Month'].map(season_map)
    med_month = df.groupby(['Medication', 'Season'], observed=True)['Sales'].mean().reset_index()
    heatmap_data = med_month.pivot(index='Medication', columns='Season', values='Sales')[['Winter', 'Spring', 'Summer', 'Fall']]
    fig3 = px.imshow(heatmap_data, text_auto='.1f', color_continuous_scale='YlOrRd', title='Medication Sales by Season (Heatmap)')
    region_over_time = df.groupby(['Month', 'Region'], observed=True)['Sales'].mean().reset_index()
    month_order = list(season_map.keys())
    region_over_time['Month'] = pd.Categorical(region_over_time['Month'], categories=month_order, ordered=True)
    region_over_time = region_over_time.sort_values('Month')
    fig4 = px.line(region_over_time, x='Month', y='Sales', color='Region', markers=True,
                   title='Sales Trends by Region (Line Plot)')
    allergy_df = df_eda[df_eda['Medication'] == 'Allergy Relief']
    fig5 = px.scatter(allergy_df, x='Pollen_Count', y='Sales', trendline='ols', title='Allergy Relief Sales vs Pollen Count')
    return html.Div([
    html.H2("Background and Exploratory Data Analysis"),
    html.P(
        "Accurately predicting medication demand is crucial for ensuring a steady supply of essential drugs and preventing shortages. "
        "Various factors, such as flu outbreaks, allergy seasons, and trends, influence demand across regions. "
        "By analyzing historical data and external factors, and utilizing multiple modelling approaches — including both statistical forecasting (SARIMA), "
        "machine learning models (like Linear Regression and Random Forest), and neural networks — we can better capture different aspects of demand patterns.",
        style={"margin": "10px 0", "lineHeight": "1.6", "textAlign": "left"}
    ),
        dcc.Graph(figure=fig_numeric), dcc.Graph(figure=fig_categorical),dcc.Graph(figure=fig1), 
        dcc.Graph(figure=fig2), dcc.Graph(figure=fig3),
        dcc.Graph(figure=fig4), dcc.Graph(figure=fig5),
    ])

# Add display map for html assets
model_plot_map = {
    'rf': ["rf_actual_vs_pred.html", "rf_feature_importance.html"],
    'xgb': ["xgb_actual_vs_pred.html", "xgb_feature_importance.html"],
    'nn': ["nn_actual_vs_pred.html"]
}

# Update the model metrics layout with dropdown
from dash import dash_table

def layout_model_metrics():
    metrics = {
        'XGB': {'R² Score': 0.777, 'RMSE': 215.56, 'MSE': 46464.21},
        'RandomForest': {'R² Score': 0.772, 'RMSE': 217.95, 'MSE': 47501.41},
        'DT': {'R² Score': 0.740, 'RMSE': 232.93, 'MSE': 54257.89},
        'KNN': {'R² Score': 0.554, 'RMSE': 304.78, 'MSE': 92890.29},
        'Linear': {'R² Score': 0.536, 'RMSE': 310.78, 'MSE': 96581.46},
        'AdaBoost': {'R² Score': -0.052, 'RMSE': 468.13, 'MSE': 219146.17},
        'FFNN': {'R² Score': 0.81, 'RMSE': 198.43, 'MSE': 39374.36},
    }
    df_metrics = pd.DataFrame(metrics).T.reset_index().rename(columns={'index': 'Model'})
    return html.Div([
        html.H2("Model Performance Metrics"),
        html.P("Here is the initial pool of models that we looked at to predict medication sales, along with their performance metrics when predicting medication sales.", style={"marginBottom": "10px"}),
        dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in df_metrics.columns],
            data=df_metrics.to_dict('records'),
            style_table={"overflowX": "auto", "marginTop": "10px", "marginBottom": "10px"},
            style_cell={"textAlign": "center", "padding": "6px"},
            style_header={"backgroundColor": "lightgrey", "fontWeight": "bold"},
            style_data={"whiteSpace": "normal", "height": "auto"},
        ),
        html.P("Based on performance, the models selected for tuning/further analysis included the Random Forest, XGBoost and Neural Network models.", style={"marginTop": "10px"}),

        html.Br(),
        html.Label("Select a model to view visualizations:"),
        dcc.Dropdown(
            id='model-plot-selector',
            options=[
                {'label': 'Random Forest', 'value': 'rf'},
                {'label': 'XGBoost', 'value': 'xgb'},
                {'label': 'Neural Network', 'value': 'nn'},
            ],
            placeholder='Choose a model...',
            style={"width": "400px", "margin": "10px auto"}
        ),
        html.Div(id='model-plot-container'),
        html.Div(id='model-blurb', style={"maxWidth": "800px", "margin": "20px auto", "lineHeight": "1.6", "whiteSpace": "pre-line"})
    ])

# Add callback to display HTML plots from assets
@app.callback(
    Output('model-plot-container', 'children'),
    Output('model-blurb', 'children'),
    Input('model-plot-selector', 'value')
)
def display_model_assets(model_key):
    if model_key == 'rf':
        return [
            html.P(
                "We used a Random Forest to better capture non-linear relationships and interactions between features. We maintained the same 80/20 train-test split as the linear regression model to ensure a fair comparison. The Random Forest performed significantly better, achieving an R² score of 0.77, meaning it explained about 77% of the variance in sales. The RMSE was approximately 218, indicating the model’s predictions were off by around 218 sales units on average. That’s a big improvement in both accuracy and variance explained compared to the linear regression baseline. These results suggest that Random Forest is a much stronger fit for this dataset, especially when dealing with complex patterns in the data.",
                style={"margin": "10px 0", "lineHeight": "1.6", "textAlign": "left"}
            ),
            html.Iframe(
                src="/assets/rf_actual_vs_pred.html",
                style={"width": "100%", "height": "600px", "border": "none", "marginBottom": "40px"}
            ),
            html.P("Looking at the feature importances from the Random Forest model, we found that Marketing Spend had the strongest influence on predictions,  contributing around 45% to the model’s decision-making. This highlights a potential strong relationship between marketing efforts and medication sales.  Google Trends ranked second, which suggests that online search behavior could be a useful early indicator of consumer demand.", style={"margin": "10px 0", "lineHeight": "1.6", "textAlign": "left"}),
            html.P("Interestingly, more traditional seasonal or clinical drivers like flu cases, temperature, and humidity ranked lower, indicating that consumer behaviour and marketing efforts may be more powerful predictors in this context than environmental or epidemiological factors.", style={"margin": "10px 0", "lineHeight": "1.6", "textAlign": "left"}),
            html.Iframe(
                src="/assets/rf_feature_importance.html",
                style={"width": "100%", "height": "600px", "border": "none", "marginBottom": "40px"}
            )
        ], ""

    if model_key == 'xgb':
        return [
            html.P(
                "This scatter plot shows how well the XGBoost model's predictions align with the actual sales values. Most points fall near the line, suggesting good performance overall. There is some spread at higher sales values, but the model still captures the overall trend effectively. This indicates that XGBoost is making reliable predictions across a wide range of sales values.",
                style={"margin": "10px 0", "lineHeight": "1.6", "textAlign": "left"}
            ),
            html.Iframe(
                src="/assets/xgb_actual_vs_pred.html",
                style={"width": "100%", "height": "600px", "border": "none", "marginBottom": "40px"}
            ),
            html.P("The XGBoost feature importance plot highlights Season_Winter and Marketing_Spend as the top contributors to predicting sales, followed by Medication_Cough Suppressant and Season_Summer. This suggests that seasonal context and promotional effort (spend) are highly influential in the XGBoost model.", style={"margin": "10px 0", "lineHeight": "1.6", "textAlign": "left"}),
            html.P("Comparing this to the Random Forest model, Marketing_Spend was by far the most dominant predictor, followed by Google_Trends and Medication_Cough Suppressant, with seasonal indicators like Season_Winter appearing further down.", style={"margin": "10px 0", "lineHeight": "1.6", "textAlign": "left"}),
            html.P("While both models agree on the general importance of marketing spend and certain medications, XGBoost places greater weight on seasonality, especially winter, suggesting it may be better at capturing interaction effects or nonlinear seasonal patterns in the data.", style={"margin": "10px 0", "lineHeight": "1.6", "textAlign": "left"}),
            html.Iframe(
                src="/assets/xgb_feature_importance.html",
                style={"width": "100%", "height": "600px", "border": "none", "marginBottom": "40px"}
            )
        ], ""

    if model_key == 'nn':
        return [
            html.P("""
Test MSE: 39374.36 -
This is the average of squared differences between predicted and actual sales. Because it's in squared units, it's not easy to interpret directly — 
but it's useful for optimization and comparison.

Test RMSE: 198.43 -
This is the Root Mean Squared Error, in the same units as 'Sales'. This tells us that on average, the model’s predictions are off by about 198 units of sales. 
This is easier to understand than MSE, a good metric for business or forecasting stakeholders.

RMSE as % of Average Sales: 26.77% -
This tells us how big the typical error is relative to average sales, meaning that the model’s typical prediction error is about 27% of the average sales value.

Therefore, the FFNN displayed the best performance thus far.
""", style={"margin": "10px 0", "lineHeight": "1.6", "textAlign": "left", "whiteSpace": "pre-line"}),
            html.Iframe(
                src="/assets/nn_actual_vs_pred.html",
                style={"width": "100%", "height": "600px", "border": "none", "marginBottom": "40px"}
            )
        ], ""

    return "", ""

def layout_nn_predictions():
    return html.Div([
        html.H2("Neural Network Sales Prediction"),
        html.P("Since the neural network was the best performing prediction model, feel free to try it yourself by inputting values for your medication and seeing how many sales it is projected to have. The default values already inputted below are the mean or mode (if categorical) for each variable in the model's training dataset.",
               style={'textAlign': 'left', 'marginBottom': '30px'}),
        html.Div([
            html.Div([html.Label("Temperature (°C):"), dcc.Input(id='input-temp', type='number', value=default_vals['Temperature'])]),
            html.Div([html.Label("Humidity (%):"), dcc.Input(id='input-humidity', type='number', value=default_vals['Humidity'])]),
            html.Div([html.Label("Flu Cases:"), dcc.Input(id='input-flu', type='number', value=default_vals['Flu_Cases'])]),
            html.Div([html.Label("Pollen Count:"), dcc.Input(id='input-pollen', type='number', value=default_vals['Pollen_Count'])]),
            html.Div([html.Label("Google Trends Score:"), dcc.Input(id='input-google-trends', type='number', value=default_vals['Google_Trends'])]),
            html.Div([html.Label("Marketing Spend ($):"), dcc.Input(id='input-marketing', type='number', value=default_vals['Marketing_Spend'])]),
            html.Div([html.Label("Medication Type:"),
                      dcc.Dropdown(id='input-medication', options=[{'label': m, 'value': m} for m in ['Allergy Relief', 'Cold Relief', 'Cough Suppressant', 'Fever Reducer', 'Pain Relief']], value=default_vals['Medication'])]),
            html.Div([html.Label("Region:"),
                      dcc.Dropdown(id='input-region', options=[{'label': r, 'value': r} for r in ['Calgary', 'Toronto', 'Vancouver', 'Montreal']], value=default_vals['Region'])]),
            html.Div([html.Label("Holiday:"),
                      dcc.Dropdown(id='input-holiday', options=[{'label': h, 'value': h} for h in ['Holiday', 'Not a Holiday']], value=default_vals['Holiday'])]),
            html.Div([html.Label("Season:"),
                      dcc.Dropdown(id='input-season', options=[{'label': s, 'value': s} for s in ['Fall', 'Winter', 'Spring', 'Summer']], value=default_vals['Season'])]),
            html.Br(), html.Button("Predict", id='predict-button'),
        ], style={'maxWidth': '500px', 'margin': '0 auto', 'display': 'flex', 'flexDirection': 'column', 'gap': '10px'}),
        html.Div(id='nn-prediction-output', style={'textAlign': 'center', 'marginTop': '30px', 'fontWeight': 'bold', 'fontSize': '20px'})
    ])

def layout_time_series():
    return html.Div([
        html.H2("Time Series Forecast (SARIMA)"),

        html.P(
            "Time series forecasting is a technique that uses historical data points, ordered over time, to predict future values. "
            "In this analysis, we applied time series modeling to predict sales of over-the-counter medications across regions and product types. "
            "We used the SARIMA (Seasonal AutoRegressive Integrated Moving Average) model, which accounts for trends over time (e.g., increasing or decreasing sales), "
            "seasonality (e.g., weekly or monthly patterns), and autocorrelation (how past values influence future values).",
            style={"marginBottom": "25px", "lineHeight": "1.6", "textAlign": "left"}
        ),

        html.Label("Enter number of days to forecast:"),
        dcc.Input(id='forecast-days', type='number', value=30, min=1, max=365),
        html.Br(),
        html.Button("Generate Forecast", id='forecast-button'),

        html.Div([
            html.P(
                "The model was used to forecast total sales in the future using the full historical dataset. As the future values do not exist in our dataset, "
                "we evaluated the forecast based on how well it aligns with past trends and the width of the confidence interval. The results show a stable continuation of existing seasonal patterns, "
                "with forecast uncertainty gradually increasing — which is expected in time series modeling.",
                style={"marginTop": "30px", "marginBottom": "20px", "lineHeight": "1.6", "textAlign": "left"}
            ),
            dcc.Graph(id='forecast-plot')
        ], id='ts-forecast-container', style={'display': 'none'}),

        html.Hr(style={"marginTop": "40px", "marginBottom": "30px"}),

        html.Div([
            html.P(
                "This model forecasts total daily sales across all medications and regions using a SARIMA model with weekly seasonality. "
                "To assess accuracy, the last 30 days of data were held out as a test set. Forecast performance was evaluated using MSE, RMSE, and MAE, showing how well the model predicts actual sales it was not trained on.\n"
                "MSE: 2709807.14\nRMSE: 5204.79\nMAE: 5057.70\nRelative RMSE: ~2.63% of average daily sales.\n"
                "The model performed strongly, with an RMSE of just 5,200 units per day — only 2.63% of the average daily sales volume (~14,674). "
                "This indicates that the model captured overall demand trends and weekly seasonality effectively. While some variation is expected during seasonal peaks or unexpected demand shifts, the model provides a relatively reliable short-term forecast.",
                style={"marginTop": "40px", "marginBottom": "20px", "lineHeight": "1.6", "textAlign": "left", "whiteSpace": "pre-line"}
            ),
            html.Iframe(
                src="/assets/overall_forecast_with_testset.html",
                style={"width": "100%", "height": "600px", "border": "none", "marginBottom": "40px"}
            )
        ]),

        html.Hr(style={"marginTop": "40px", "marginBottom": "20px"}),

        html.P(
            "We can break it down further by region and medication, please select the combination you would like to see along with the forecast accuracy (RMSE) for each combination.",
            style={"marginBottom": "20px", "lineHeight": "1.6", "textAlign": "left"}
        ),

        html.Div([
            html.Label("Select Region:"),
            dcc.Dropdown(
                id='region-dropdown',
                options=[{'label': r, 'value': r} for r in sorted(df['Region'].unique())],
                style={'marginBottom': '20px'}
            ),

            html.Label("Select Medication:"),
            dcc.Dropdown(
                id='medication-dropdown',
                options=[{'label': m, 'value': m} for m in sorted(df['Medication'].unique())],
                style={'marginBottom': '20px'}
            ),

            dcc.Graph(id='region-medication-forecast')
        ], style={"marginTop": "20px"})
    ])


# ----------------- App Layout -----------------

app.layout = html.Div([
    dcc.Tabs(id="tabs", value='tab-eda', children=[
        dcc.Tab(label='Background & EDA', value='tab-eda'),
        dcc.Tab(label='Model Metrics', value='tab-metrics'),
        dcc.Tab(label='Neural Network Predictions', value='tab-nn'),
        dcc.Tab(label='Time Series Forecast', value='tab-ts'),
    ]),
    html.Div(id='tabs-content')
])

@app.callback(Output('tabs-content', 'children'), Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-eda': return layout_background_eda()
    if tab == 'tab-metrics': return layout_model_metrics()
    if tab == 'tab-nn': return layout_nn_predictions()
    if tab == 'tab-ts': return layout_time_series()


# ----------------- Callbacks -----------------

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
        'FFNN': {'R² Score': 0.81, 'RMSE': 198.43, 'MSE': 39374.36},
    }
    m = metrics[model_name]
    return html.Ul([html.Li(f"{k}: {v}") for k, v in m.items()])

@app.callback(
    Output('nn-prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    State('input-temp', 'value'), State('input-humidity', 'value'), State('input-flu', 'value'),
    State('input-pollen', 'value'), State('input-google-trends', 'value'), State('input-marketing', 'value'),
    State('input-medication', 'value'), State('input-region', 'value'),
    State('input-holiday', 'value'), State('input-season', 'value')
)
def predict_sales(n_clicks, temp, humidity, flu, pollen, trends, marketing, medication, region, holiday, season):
    if not n_clicks:
        return ""
    if any(v is None for v in [temp, humidity, flu, pollen, trends, marketing, medication, region, holiday, season]):
        return "⚠️ Please fill in all input fields before predicting."
    raw_input = {
        'Temperature': temp, 'Humidity': humidity, 'Flu_Cases': flu,
        'Pollen_Count': pollen, 'Google_Trends': trends, 'Marketing_Spend': marketing,
        'Medication': medication, 'Region': region, 'Holiday': holiday, 'Season': season
    }
    df_input = pd.DataFrame([raw_input])
    df_input = pd.get_dummies(df_input)
    for col in input_columns:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[input_columns]
    scaled_input = scaler.transform(df_input)
    prediction = nn.predict(scaled_input)[0][0]
    return f"Predicted Sales: {prediction:.2f} units"

# Add callback to show the container when button clicked
@app.callback(
    Output('ts-forecast-container', 'style'),
    Input('forecast-button', 'n_clicks')
)
def show_forecast_container(n_clicks):
    if n_clicks:
        return {'display': 'block'}
    return {'display': 'none'}

# Modify existing forecast callback to update forecast-plot directly
@app.callback(
    Output('forecast-plot', 'figure'),
    Input('forecast-button', 'n_clicks'),
    State('forecast-days', 'value')
)
def update_forecast_figure(n_clicks, days):
    if not n_clicks or not days:
        return go.Figure().update_layout(
            title="Click 'Generate Forecast' to display the forecast",
            xaxis_title="Date",
            yaxis_title="Sales",
            template="plotly_white"
        )

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
        fill='toself', fillcolor='rgba(255,165,0,0.3)', line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip", name='Confidence Interval'))

    fig.update_layout(
        title=f"Sales Forecast for Next {days} Days",
        xaxis_title="Date", yaxis_title="Sales", template="plotly_white")

    return fig

# Region-Medication Forecast — show blank until input
@app.callback(
    Output('region-medication-forecast', 'figure'),
    Input('region-dropdown', 'value'),
    Input('medication-dropdown', 'value')
)
def update_region_medication_forecast(region, medication):
    if not region or not medication:
        return go.Figure()  # Blank figure until inputs are provided

    df_filtered = df[(df['Medication'] == medication) & (df['Region'] == region)].copy()
    df_filtered['Date'] = pd.to_datetime(df_filtered['Date'])
    df_grouped = df_filtered.groupby('Date')['Sales'].sum().asfreq('D').ffill()

    if df_grouped.isna().sum() > 0 or len(df_grouped) < 60:
        return go.Figure()  # Blank instead of error message

    train = df_grouped[:-30]
    test = df_grouped[-30:]

    try:
        model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
        results = model.fit(disp=False)
        forecast = results.get_forecast(steps=30)
        forecast_index = pd.date_range(start=train.index[-1] + timedelta(days=1), periods=30)
        forecast_series = pd.Series(forecast.predicted_mean.values, index=forecast_index)
        conf_int = forecast.conf_int()

        test = test.reindex(forecast_index)

        mse = mean_squared_error(test, forecast_series)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test, forecast_series)

        interpretation_lookup = {
            ("Cold Relief", "Toronto"): "Forecasting was challenging due to sharp winter demand spikes (RMSE: 849).",
            ("Cold Relief", "Vancouver"): "Moderate accuracy; fewer extreme seasonal peaks contributed to lower error (RMSE: 548).",
            ("Cold Relief", "Montreal"): "Frequent cold-season surges led to high forecast error (RMSE: 879).",
            ("Cold Relief", "Calgary"): "Forecasts were moderately accurate; demand fluctuated during winter months (RMSE: 798).",
            ("Allergy Relief", "Toronto"): "Forecasts were highly accurate, supported by consistent spring allergy patterns (RMSE: 75).",
            ("Allergy Relief", "Vancouver"): "Spring-related demand was well captured, resulting in low error (RMSE: 109).",
            ("Allergy Relief", "Montreal"): "Slightly higher variability in allergy season led to moderate forecast error (RMSE: 155).",
            ("Allergy Relief", "Calgary"): "Clear seasonal trends supported strong model performance (RMSE: 82).",
            ("Pain Relief", "Toronto"): "Mild fluctuations in daily use contributed to modest error (RMSE: 268).",
            ("Pain Relief", "Vancouver"): "Stable demand throughout the year allowed for reliable forecasts (RMSE: 134).",
            ("Pain Relief", "Montreal"): "Minimal seasonality supported accurate model results (RMSE: 127).",
            ("Pain Relief", "Calgary"): "Very strong performance driven by steady usage patterns (RMSE: 107).",
            ("Cough Suppressant", "Toronto"): "Unpredictable seasonal spikes resulted in the highest forecast error (RMSE: 1011).",
            ("Cough Suppressant", "Vancouver"): "Moderate accuracy; seasonal variation contributed to forecast noise (RMSE: 510).",
            ("Cough Suppressant", "Montreal"): "Cold-season demand was variable, reducing forecast reliability (RMSE: 551).",
            ("Cough Suppressant", "Calgary"): "Lower error compared to other regions; seasonal shifts were more predictable (RMSE: 324).",
            ("Fever Reducer", "Toronto"): "Forecasts were affected by flu season peaks, particularly in winter (RMSE: 526).",
            ("Fever Reducer", "Vancouver"): "Low forecast error due to relatively stable demand (RMSE: 226).",
            ("Fever Reducer", "Montreal"): "Moderate accuracy; some demand variability during peak illness periods (RMSE: 427).",
            ("Fever Reducer", "Calgary"): "Consistent performance; mild seasonal influence captured effectively (RMSE: 387)."
        }
        interpretation = interpretation_lookup.get((medication, region), "Interpretation not available.")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train.index, y=train.values, mode='lines', name='Train'))
        fig.add_trace(go.Scatter(x=test.index, y=test.values, mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(x=forecast_index, y=forecast_series, mode='lines', name='Forecast'))
        fig.add_trace(go.Scatter(
            x=forecast_index.tolist() + forecast_index[::-1].tolist(),
            y=conf_int.iloc[:, 0].tolist() + conf_int.iloc[:, 1][::-1].tolist(),
            fill='toself', fillcolor='rgba(255,165,0,0.3)', line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip", name='Confidence Interval'))

        fig.update_layout(
            title=f"{medication} Sales Forecast - {region}<br><sub>{interpretation}</sub>",
            xaxis_title="Date", yaxis_title="Sales", template="plotly_white")

        return fig

    except Exception as e:
        return go.Figure()


# ----------------- Run Server -----------------

if __name__ == '__main__':
    app.run(debug=True)