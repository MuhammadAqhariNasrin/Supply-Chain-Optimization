import pickle
import pandas as pd
from fastapi import FastAPI, Form
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Supply CHain Optimization API!"}

# Load the model and scaler
regmodel = pickle.load(open('TransformedTargetRegressor_model.pkl', 'rb'))
scalar = pickle.load(open('scaler.pkl', 'rb'))

# Define a Pydantic model to handle input data
class InputData(BaseModel):
    product_type: str
    sku: str
    price: float
    availability: int
    num_products_sold: int
    revenue_generated: float
    customer_demographics: str
    stock_levels: int
    lead_times: int
    order_quantities: int
    shipping_times: int
    shipping_carriers: str
    shipping_costs: float
    supplier_name: str
    location: str
    lead_time: int
    production_volumes: int
    manufacturing_lead_time: int
    manufacturing_costs: float
    inspection_results: str
    defect_rates: float
    transportation_modes: str
    routes: str
    costs: float

@app.post('/predict')
def predict(data: InputData):
    # Create a DataFrame from the input data
    input_df = pd.DataFrame(data.dict(), index=[0])

    # List of columns to one-hot encode
    columns_to_encode = ['product_type', 'customer_demographics', 'shipping_carriers', 'supplier_name', 'location', 'transportation_modes', 'inspection_results', 'routes']

    # Perform one-hot encoding on the specified columns
    input_df_encoded = pd.get_dummies(input_df, columns=columns_to_encode)

    # Ensure that the input data has the same columns as the model was trained on
    expected_columns = set([
        "price",
        "availability",
        "num_products_sold",
        "revenue_generated",
        "stock_levels",
        "lead_times",
        "order_quantities",
        "shipping_times",
        "shipping_costs",
        "lead_time",
        "production_volumes",
        "manufacturing_lead_time",
        "manufacturing_costs",
        "defect_rates",
        "supplier_score",
        "costs",
    ])

    # Iterate over the one-hot encoded columns and set to 1 or 0 based on user input
    for col in columns_to_encode:
        if col in input_df.columns:
            unique_values = input_df[col].unique()
            for unique_value in unique_values:
                input_df_encoded[f"{col}_{unique_value}"] = (input_df[col] == unique_value).astype(int)

    # Drop the original categorical columns if they exist in the DataFrame
    input_df_encoded = input_df_encoded.drop(columns=[col for col in columns_to_encode if col in input_df_encoded])

    missing_columns = expected_columns - set(input_df_encoded.columns)
    if missing_columns:
        return {"error": f"Missing columns: {missing_columns}"}

    # Ensure that there are no extra columns in the input data
    extra_columns = set(input_df_encoded.columns) - expected_columns
    if extra_columns:
        return {"error": f"Extra columns found: {extra_columns}"}

    # Perform standard scaling using the scalar
    preprocessed_data = scalar.transform(input_df_encoded)

    # Make predictions using the model
    prediction = regmodel.predict(preprocessed_data)

    return {'prediction': prediction[0]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


# Load the model and scaler
regmodel = pickle.load(open('TransformedTargetRegressor_model.pkl', 'rb'))
scalar = pickle.load(open('scaler.pkl', 'rb'))

# Define a Pydantic model to handle input data
class InputData(BaseModel):
    product_type: str
    sku: str
    price: float
    availability: int
    num_products_sold: int
    revenue_generated: float
    customer_demographics: str
    stock_levels: int
    lead_times: int
    order_quantities: int
    shipping_times: int
    shipping_carriers: str
    shipping_costs: float
    supplier_name: str
    location: str
    lead_time: int
    production_volumes: int
    manufacturing_lead_time: int
    manufacturing_costs: float
    inspection_results: str
    defect_rates: float
    transportation_modes: str
    routes: str
    costs: float

@app.post('/predict')
def predict(data: InputData):
    # Create a DataFrame from the input data
    input_df = pd.DataFrame(data.dict(), index=[0])

    # List of columns to one-hot encode
    columns_to_encode = ['product_type', 'customer_demographics', 'shipping_carriers', 'supplier_name', 'location', 'transportation_modes', 'inspection_results', 'routes']

    # Perform one-hot encoding on the specified columns
    input_df_encoded = pd.get_dummies(input_df, columns=columns_to_encode)

    # Ensure that the input data has the same columns as the model was trained on
    expected_columns = set([
        "price",
        "availability",
        "num_products_sold",
        "revenue_generated",
        "stock_levels",
        "lead_times",
        "order_quantities",
        "shipping_times",
        "shipping_costs",
        "lead_time",
        "production_volumes",
        "manufacturing_lead_time",
        "manufacturing_costs",
        "defect_rates",
        
        "costs",
    ])

    # Iterate over the one-hot encoded columns and set to 1 or 0 based on user input
    for col in columns_to_encode:
        if col in input_df.columns:
            unique_values = input_df[col].unique()
            for unique_value in unique_values:
                input_df_encoded[f"{col}_{unique_value}"] = (input_df[col] == unique_value).astype(int)

    # Drop the original categorical columns if they exist in the DataFrame
    input_df_encoded = input_df_encoded.drop(columns=[col for col in columns_to_encode if col in input_df_encoded])

    missing_columns = expected_columns - set(input_df_encoded.columns)
    if missing_columns:
        return {"error": f"Missing columns: {missing_columns}"}

    # Ensure that there are no extra columns in the input data
    extra_columns = set(input_df_encoded.columns) - expected_columns
    if extra_columns:
        return {"error": f"Extra columns found: {extra_columns}"}

    # Perform standard scaling using the scalar
    preprocessed_data = scalar.transform(input_df_encoded)

    # Make predictions using the model
    prediction = regmodel.predict(preprocessed_data)

    return {'prediction': prediction[0]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)