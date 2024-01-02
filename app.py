import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import requests
import os
import json
from PIL import Image
from io import BytesIO
import base64
import io
import plotly.graph_objs as go



# Set page config
st.set_page_config(page_title="Semantic Product Search", page_icon="🌐", layout="wide")

# Include DM Sans font in the entire app
st.markdown(
    """
    <head>
        <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap" rel="stylesheet">
    </head>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    html, body, h1, h2, h3, h4, h5, h6, p, div, span, a, li, button, input, select, textarea {
        font-family: 'DM Sans', sans-serif !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)



os.environ['DATABRICKS_TOKEN'] = '<your_token>'
lookup_dict = pd.read_csv('fashion.csv').set_index('Image').to_dict(orient='index')
prod_df= pd.read_csv("prod_stats.csv")


def plot_sns(prodid):
    # Set the aesthetic style of the plots
    sns.set_style("whitegrid")

    # Create a figure and a set of subplots
    plt.figure(figsize=(12, 8))

    # Plotting each dependent variable against 'Week'
    sns.lineplot(data=df2, x='Week', y='units_sold', label='Units Sold')
    sns.lineplot(data=df2, x='Week', y='units_onhand', label='Units On Hand')
    sns.lineplot(data=df2, x='Week', y='OOS', label='OOS') # Assuming OOS is a numerical column

    # Adding title and labels
    plt.title('Weekly Data Trends for Product 1636')
    plt.xlabel('Week')
    plt.ylabel('Values')
    plt.legend(title='Variables')

    # Show the plot
    plt.show()


# Initialize session state variables
if 'results' not in st.session_state:
    st.session_state['results'] = None
if 'lookup_dict' not in st.session_state:
    st.session_state['lookup_dict'] = lookup_dict

# Define scoring model function
def score_model(model_input):
    dataset = pd.DataFrame([[model_input]], columns=['text'])
    url = 'https://adb-4522126718558142.2.azuredatabricks.net/serving-endpoints/nrfbot/invocations'
    headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}',
               'Content-Type': 'application/json'}
    ds_dict = {'dataframe_split': dataset.to_dict(orient='split')}
    data_json = json.dumps(ds_dict, allow_nan=True)
    response = requests.request(method='POST', headers=headers, url=url, data=data_json)
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    return response.json()['predictions']

# Custom CSS
def local_css(file_name):
    with open(file_name, "r") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

st.markdown(
    """
    <head>
        <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap" rel="stylesheet">
    </head>
    <style>
    h1 {
        font-family: 'DM Sans', sans-serif;
        font-weight: 500; /* Medium weight */
        font-size: 42px; /* Adjust the size as needed */
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Logo and Title
col1, col2 = st.columns([1, 6])
with col1:
    st.image("logo2.png", width=70)
with col2:
    st.title("Semantic Product Search")

# File Upload, Search Input, and Search Button

search_input = st.text_input(" ", key="search")
search_button = st.button("Search")


# Handling the search button
if search_button:
    st.write(f"Search for: {search_input}")
    st.session_state['results'] = json.loads(score_model(search_input))


# Check if a search has been performed (search input is not empty or results are available)
if search_input or (st.session_state.get('results') is not None):
    # Display results and hidden expanders for smaller plots
# Display results and hidden expanders for smaller plots
    for i in range(3):
        with st.expander(f"Item {i+1}", expanded=True):
            col_img, col_desc = st.columns([1, 4])  # Adjusted column sizes

            with col_img:
                if st.session_state['results'] is not None:
                    img_str = st.session_state['results']['bs64_images'][i]
                    image_string = base64.b64decode(img_str)
                    image_string = Image.open(BytesIO(image_string)).convert('RGB')
                    st.image(image_string, width=150)
                else:
                    st.image("placeholder.svg", width=150)

            with col_desc:
                if st.session_state['results'] is not None: 
                    # st.write(st.session_state['results']['images'][i])
                    # st.write("**Product Details**")
                    # keys = list(st.session_state['lookup_dict'][st.session_state['results']['images'][i]].keys())[:-1]
                    # for key in keys:
                    #     st.text(f"{key}: {st.session_state['lookup_dict'][st.session_state['results']['images'][i]][key]}")

                    # Retrieve product details
                    product_details = st.session_state['lookup_dict'][st.session_state['results']['images'][i]]

                    # # Display formatted product details
                    # st.markdown(f"**Product Title ({product_details['ProductId']}):**")
                    # st.markdown(f"**Type:** {product_details['ProductType']} \t **Color:** {product_details['Colour']}")
                    # st.markdown(f"**Category:** {product_details['Category']} \t **Gender:** {product_details['Gender']}")
                    # st.markdown(f"**Subcategory:** {product_details['SubCategory']} \t **Usage:** {product_details['Usage']}")
                    # Display formatted product details in two columns
                    st.markdown(f"**Product Title ({product_details['ProductId']}):**")
                    
                    col_left, col_right = st.columns(2)
                    with col_left:
                        st.text(f"Type: {product_details['ProductType']}")
                        st.text(f"Category: {product_details['Category']}")
                        st.text(f"Subcategory: {product_details['SubCategory']}")
                    with col_right:
                        st.text(f"Color: {product_details['Colour']}")
                        st.text(f"Gender: {product_details['Gender']}")
                        st.text(f"Usage: {product_details['Usage']}")

                else:
                    st.write("Product Details")

    # Displa# Forecast Information
            # Forecast Information
            # Plotting Units Sold and On Hand Combined
            # Plotting Units Sold and On Hand Combined
            # Plotting Units Sold and On Hand Combined
            if st.session_state['results'] is not None:
                prodid = st.session_state['results']['images'][i].split(".")[0]
                df2 = prod_df[prod_df["ProductId"]==int(prodid)]

                # Create a Plotly figure
                fig = go.Figure()

                # Separate the data based on 'status'
                actuals = df2[df2['status'] == 'actuals']
                forecast = df2[df2['status'] == 'forecast']

                # Add traces for actuals and forecast
                fig.add_trace(go.Scatter(x=actuals['Week'], y=actuals['units_sold'], mode='lines', name='Units Sold (Actuals)', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=actuals['Week'], y=actuals['units_onhand'], mode='lines', name='Units On Hand (Actuals)', line=dict(color='green')))
                fig.add_trace(go.Scatter(x=forecast['Week'], y=forecast['units_sold'], mode='lines', name='Units Sold (Forecast)', line=dict(color='cyan')))
                fig.add_trace(go.Scatter(x=forecast['Week'], y=forecast['units_onhand'], mode='lines', name='Units On Hand (Forecast)', line=dict(color='lime')))

                # Mark OOS events and add a representative trace for the legend
                oos_events = df2[df2['OOS'] == 1.0]
                if not oos_events.empty:
                    fig.add_trace(go.Scatter(x=[oos_events.iloc[0]['Week']], y=[oos_events.iloc[0]['units_sold']], mode='markers', marker=dict(color='red', size=10), name='Out of Stock'))

                for _, event in oos_events.iterrows():
                    #fig.add_trace(go.Scatter(x=[event['Week']], y=[event['units_sold']], mode='markers', marker=dict(color='red', size=10), showlegend=False))
                    fig.add_trace(go.Scatter(x=[event['Week']], y=[event['units_onhand']], mode='markers', marker=dict(color='red', size=10), showlegend=False))

                # Update layout
                fig.update_layout(title=f'Weekly Data Trends for Product {prodid}', xaxis_title='Week', yaxis_title='Values')

                # Display Plotly figure in Streamlit
                st.plotly_chart(fig, use_container_width=False)
            else:
                st.write("No forecast data available.")