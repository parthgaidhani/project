import os
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from flask import Flask, request, render_template
import base64
import io

# Initialize Flask app
server = Flask(__name__)

# Read the logo image and encode it as a base64 string
logo_path = r"C:\Users\kisho\Downloads\zbtc58xk.png"
with open(logo_path, "rb") as image_file:
    encoded_logo = base64.b64encode(image_file.read()).decode("utf-8")

# Initialize Dash app within the Flask app
app = dash.Dash(__name__, server=server)

# Define the layout of the Dash app with a logo
app.layout = html.Div([
    html.Div([
        html.H1("Welcome to SoftGrow App", style={'textAlign': 'center', 'color': '#333'}),
        html.Img(src='data:image/png;base64,{}'.format(encoded_logo),
                 height=80, style={'border-radius': '50%', 'margin-right': '10px'}),
    ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}),
    html.P("Explore your data visually with various chart types.", style={'textAlign': 'center', 'color': '#333'}),
    dcc.Location(id='url', refresh=False),
    html.Nav([
        dcc.Link('Home', href='/', style={'color': '#333', 'margin': '0 10px'}),
        dcc.Link('About', href='/about', style={'color': '#333', 'margin': '0 10px'}),
        dcc.Link('Upload Data', href='/upload-data', style={'color': '#333', 'margin': '0 10px'}),
        dcc.Link('Payment', href='/payment', style={'color': '#333', 'margin': '0 10px'}),
    ], style={'textAlign': 'center', 'margin-top': '20px'}),
    html.Div(id='page-content', style={'padding': '20px'}),
], style={'maxWidth': '800px', 'margin': 'auto', 'font-family': 'Arial, sans-serif', 'color': '#333'})

# Home Page
home_layout = html.Div([
    html.Div([
        html.H1("Welcome to SoftGrow App", style={'textAlign': 'center', 'color': '#333'}),
        html.Img(src='data:image/png;base64,{}'.format(encoded_logo),
                 height=80, style={'border-radius': '50%', 'margin-left': '230px'}),
    ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}),
    html.P("Explore your data visually with various chart types.", style={'textAlign': 'center', 'color': '#333'}),
])

# About Page
about_layout = html.Div([
    html.H1("About SoftGrow App", style={'textAlign': 'center', 'color': '#333'}),
    html.P("This web application allows you to upload a CSV file and visualize the data using different chart types."),
])

# Upload Data Page
upload_data_layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select a CSV File', style={'color': '#3498db', 'textDecoration': 'underline'}),
        ]),
        multiple=False,
        style={'marginTop': '20px', 'marginBottom': '20px', 'border': '2px dashed #3498db', 'border-radius': '5px',
               'padding': '20px', 'text-align': 'center'}
    ),
    dcc.Dropdown(
        id='graph-type-dropdown',
        options=[
            {'label': 'Scatter Plot', 'value': 'scatter'},
            {'label': 'Line Plot', 'value': 'line'},
            {'label': 'Histogram', 'value': 'histogram'},
            {'label': 'Area Plot', 'value': 'area'},
            {'label': 'Pie Chart', 'value': 'pie'},
            {'label': 'Dual-Axis ', 'value': 'dual-axis'},
            {'label': 'time-series', 'value': 'time-series'},
        ],
        value='scatter',
        style={'width': '50%', 'marginBottom': '20px'}
    ),
    dcc.Graph(id='data-visualization'),
])

# Payment Page
payment_layout = html.Div([
    html.H1(" Payment ", style={'textAlign': 'center', 'color': '#333'}),
    html.P("Enter your payment details below:"),
    html.Div([
        html.Label('Card Number:'),
        dcc.Input(id='card-number', type='text', placeholder='Enter card number',
                  style={'width': '100%', 'marginBottom': '10px'}),
        html.Label('Expiration Date:'),
        dcc.Input(id='expiration-date', type='text', placeholder='MM/YY', style={'width': '100%', 'marginBottom': '10px'}),
        html.Label('CVV:'),
        dcc.Input(id='cvv', type='text', placeholder='Enter CVV', style={'width': '100%', 'marginBottom': '20px'}),
        html.Button('Make Payment', id='payment-button', n_clicks=0,
                    style={'background-color': '#3498db', 'color': '#fff', 'padding': '10px 20px', 'border': 'none',
                           'cursor': 'pointer'}),
        html.Div(id='payment-status', style={'marginTop': '20px', 'font-weight': 'bold'}),
    ], style={'maxWidth': '400px', 'margin': 'auto', 'text-align': 'center'}),
])


# Callback to update graph based on user input
@app.callback(
    Output('data-visualization', 'figure'),
    [Input('upload-data', 'contents'),
     Input('graph-type-dropdown', 'value')],
    [State('url', 'pathname')]
)
def update_graph(contents, graph_type, pathname):
    if pathname != '/upload-data':
        return {}

    if contents is None:
        return {}

    # Decode and read the CSV file
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

    # Create the selected type of plot
    if graph_type == 'scatter':
        fig = {
            'data': [
                {'x': df.index, 'y': df[column], 'mode': 'markers', 'name': column}
                for column in df.columns
            ],
            'layout': {
                'title': 'Scatter Plot',
                'xaxis': {'title': 'Index'},
                'yaxis': {'title': 'Values'},
                'hovermode': 'closest'
            }
        }
    elif graph_type == 'line':
        fig = {
            'data': [
                {'x': df.index, 'y': df[column], 'type': 'line', 'name': column}
                for column in df.columns
            ],
            'layout': {
                'title': 'Line Plot',
                'xaxis': {'title': 'Index'},
                'yaxis': {'title': 'Values'},
                'hovermode': 'closest'
            }
        }
    elif graph_type == 'histogram':
        fig = {
            'data': [
                {'x': df[column], 'type': 'histogram', 'name': column}
                for column in df.columns
            ],
            'layout': {
                'title': 'Histogram',
                'xaxis': {'title': 'Values'},
                'yaxis': {'title': 'Frequency'},
                'bargap': 0.1
            }
        }
    elif graph_type == 'area':
        fig = {
            'data': [
                {'x': df.index, 'y': df[column], 'type': 'scatter', 'mode': 'lines', 'fill': 'tozeroy', 'name': column}
                for column in df.columns
            ],
            'layout': {
                'title': 'Area Plot',
                'xaxis': {'title': 'Index'},
                'yaxis': {'title': 'Values'},
                'hovermode': 'closest'
            }
        }
    elif graph_type == 'pie':
        fig = {
            'data': [
                {'labels': df.index, 'parents': [column] * len(df.index), 'values': df[column], 'type': 'sunburst',
                 'name': column}
                for column in df.columns
            ],
            'layout': {
                'title': 'Sunburst Chart',
                'margin': dict(l=0, r=0, b=0, t=40)
            }
        }
    elif graph_type == 'dual-axis':
        fig = {
            'data': [
                {'x': df.index, 'y': df[column], 'name': column}
                for column in df.columns
            ],
            'layout': {
                'title': 'Dual-Axis Chart',
                'xaxis': {'title': 'Index'},
                'yaxis': {'title': 'Primary Y-Axis'},
                'yaxis2': {'title': 'Secondary Y-Axis', 'overlaying': 'y', 'side': 'right'},
                'hovermode': 'closest'
            }
        }
    elif graph_type == 'time-series':
        fig = {
            'data': [
                {'x': pd.to_datetime(df.index), 'y': df[column], 'name': column}
                for column in df.columns
            ],
            'layout': {
                'title': 'Time Series Chart',
                'xaxis': {'title': 'Time'},
                'yaxis': {'title': 'Values'},
                'hovermode': 'closest'
            }
        }

    return fig


# Callback to update page content based on URL
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/about':
        return about_layout
    elif pathname == '/upload-data':
        return upload_data_layout
    elif pathname == '/payment':
        return payment_layout
    else:
        return home_layout


# Callback for payment button click
@app.callback(
    Output('payment-status', 'children'),
    [Input('payment-button', 'n_clicks')],
    [State('card-number', 'value'),
     State('expiration-date', 'value'),
     State('cvv', 'value')]
)
def make_payment(n_clicks, card_number, expiration_date, cvv):
    if n_clicks > 0:
        # Perform payment processing here (placeholder)
        payment_successful = True  # Replace with actual payment processing logic

        if payment_successful:
            return html.Div('Payment successful!', style={'color': 'green'})
        else:
            return html.Div('Payment failed. Please try again.', style={'color': 'red'})
    return ''


# Define a route to render the HTML template
@server.route('/')
def index():
    return render_template('index.html', logo=encoded_logo)


# Define a route for uploading files
@server.route('/upload', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        file_path = os.path.join('uploads', uploaded_file.filename)
        uploaded_file.save(file_path)
        return 'File uploaded successfully!'
    return 'File not uploaded.'

# Run the application
if __name__ == '__main__':
    server.run(debug=True)
