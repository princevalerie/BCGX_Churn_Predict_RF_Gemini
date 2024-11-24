import streamlit as st
import pandas as pd
import joblib
import os
import google.generativeai as genai
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Configure Gemini API using the key from .env
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    st.error("GEMINI_API_KEY is missing in your environment variables.")
genai.configure(api_key=gemini_api_key)

gemini_api_key = os.getenv("GEMINI_API_KEY")
# if not gemini_api_key:
#     st.error("GEMINI_API_KEY is missing or not loaded correctly.")
# else:
#     st.write(f"GEMINI_API_KEY loaded: {gemini_api_key[:5]}******")

# Load Model
@st.cache_resource
def load_model():
    return joblib.load('rf_classifier_model.pkl')

# Streamlit App Title
st.title("Customer Churn Prediction App")

# Input Features
st.header("Input Features")

# Semua fitur sebagai input utama (pastikan semua angka dalam tipe float)
features = {
    'cons_12m': st.number_input('Consumption 12m', 
                                min_value=0.0, 
                                max_value=float(6_207_104 * 3),  # 3x lipat
                                value=54946.0,  # Nilai default pelanggan
                                help="Konsumsi dalam 12 bulan terakhir (max: 6,207,104)"),
    'cons_gas_12m': st.number_input('Gas Consumption 12m', 
                                    min_value=0.0, 
                                    max_value=float(4_154_590 * 3),  # 3x lipat
                                    value=0.0,  # Nilai default pelanggan
                                    help="Konsumsi gas dalam 12 bulan terakhir (max: 4,154,590)"),
    'cons_last_month': st.number_input('Consumption Last Month', 
                                       min_value=0.0, 
                                       max_value=float(771_203 * 3),  # 3x lipat
                                       value=0.0,  # Nilai default pelanggan
                                       help="Konsumsi bulan terakhir (max: 771,203)"),
    'forecast_meter_rent_12m': st.number_input('Forecast Meter Rent 12m', 
                                               min_value=0.0, 
                                               max_value=float(599.31 * 3),  # 3x lipat
                                               value=1.78,  # Nilai default pelanggan
                                               step=0.01, 
                                               help="Perkiraan sewa meter 12 bulan (max: 599.31)"),
    'forecast_price_energy_peak': st.number_input('Forecast Energy Price Peak', 
                                                  min_value=0.0, 
                                                  max_value=float(0.195975 * 3),  # 3x lipat
                                                  value=0.098142,  # Nilai default pelanggan
                                                  step=0.001, 
                                                  help="Perkiraan harga energi puncak (max: 0.195975)"),
    'margin_gross_pow_ele': st.number_input('Gross Margin Power Ele', 
                                            min_value=0.0, 
                                            max_value=float(374.64 * 3),  # 3x lipat
                                            value=25.44,  # Nilai default pelanggan
                                            step=0.01, 
                                            help="Margin kotor daya listrik (max: 374.64)"),
    'margin_net_pow_ele': st.number_input('Net Margin Power Ele', 
                                          min_value=0.0, 
                                          max_value=float(374.64 * 3),  # 3x lipat
                                          value=25.44,  # Nilai default pelanggan
                                          step=0.01, 
                                          help="Margin bersih daya listrik (max: 374.64)"),
    'nb_prod_act': st.number_input('Number of Active Products', 
                                   min_value=0.0,  # Pastikan float
                                   max_value=float(32 * 3),  # 3x lipat
                                   value=2.0,  # Nilai default pelanggan
                                   help="Jumlah produk aktif (max: 32)"),
    'net_margin': st.number_input('Net Margin', 
                                  min_value=0.0, 
                                  max_value=float(24_570 * 3),  # 3x lipat
                                  value=678.99,  # Nilai default pelanggan
                                  help="Margin bersih (max: 24,570)"),
    'num_years_antig': st.number_input('Number of Years', 
                                       min_value=0.0,  # Pastikan float
                                       max_value=float(13 * 3),  # 3x lipat
                                       value=3.0,  # Nilai default pelanggan
                                       help="Jumlah tahun (max: 13)"),
    'pow_max': st.number_input('Max Power', 
                               min_value=0.0,  # Pastikan float
                               max_value=float(320.0 * 3),  # 3x lipat
                               value=43.648,  # Nilai default pelanggan
                               step=0.1, 
                               help="Daya maksimum (max: 320.0)"),
    'avg_price_peak_var': st.number_input('Avg Price Peak Var', 
                                          min_value=0.0, 
                                          max_value=float(0.196 * 3),  # 3x lipat
                                          value=0.100749,  # Nilai default pelanggan
                                          step=0.001, 
                                          help="Rata-rata harga variabel puncak (max: 0.196)"),
    'avg_price_mid_peak_var': st.number_input('Avg Price Mid Peak Var', 
                                              min_value=0.0, 
                                              max_value=float(0.102951 * 3),  # 3x lipat
                                              value=0.066530,  # Nilai default pelanggan
                                              step=0.001, 
                                              help="Rata-rata harga variabel mid peak (max: 0.102951)"),
    'avg_price_off_peak_fix': st.number_input('Avg Price Off Peak Fix', 
                                              min_value=0.0, 
                                              max_value=float(59.286 * 3),  # 3x lipat
                                              value=40.942265,  # Nilai default pelanggan
                                              step=0.01, 
                                              help="Rata-rata harga tetap off peak (max: 59.286)"),
    'avg_price_peak_fix': st.number_input('Avg Price Peak Fix', 
                                          min_value=0.0, 
                                          max_value=float(36.490689 * 3),  # 3x lipat
                                          value=22.352010,  # Nilai default pelanggan
                                          step=0.01, 
                                          help="Rata-rata harga tetap puncak (max: 36.490689)"),
    'avg_price_mid_peak_fix': st.number_input('Avg Price Mid Peak Fix', 
                                              min_value=0.0, 
                                              max_value=float(16.818917 * 3),  # 3x lipat
                                              value=14.901340,  # Nilai default pelanggan
                                              step=0.01, 
                                              help="Rata-rata harga tetap mid peak (max: 16.818917)"),
    'max_price_peak_var': st.number_input('Max Price Peak Var', 
                                          min_value=0.0, 
                                          max_value=float(0.229788 * 3),  # 3x lipat
                                          value=0.103963,  # Nilai default pelanggan
                                          step=0.001, 
                                          help="Harga maksimum variabel puncak (max: 0.229788)"),
    'max_price_mid_peak_var': st.number_input('Max Price Mid Peak Var', 
                                              min_value=0.0, 
                                              max_value=float(0.114102 * 3),  # 3x lipat
                                              value=0.073873,  # Nilai default pelanggan
                                              step=0.001, 
                                              help="Harga maksimum variabel mid peak (max: 0.114102)"),
    'max_price_off_peak_fix': st.number_input('Max Price Off Peak Fix', 
                                              min_value=0.0, 
                                              max_value=float(59.444710 * 3),  # 3x lipat
                                              value=44.266930,  # Nilai default pelanggan
                                              step=0.01, 
                                              help="Harga maksimum tetap off peak (max: 59.444710)"),
    'max_price_peak_fix': st.number_input('Max Price Peak Fix', 
                                          min_value=0.0, 
                                          max_value=float(36.490689 * 3),  # 3x lipat
                                          value=24.43733,  # Nilai default pelanggan
                                          step=0.01, 
                                          help="Harga maksimum tetap puncak (max: 36.490689)"),
    'max_price_mid_peak_fix': st.number_input('Max Price Mid Peak Fix', 
                                              min_value=0.0, 
                                              max_value=float(17.458221 * 3),  # 3x lipat
                                              value=16.291555,  # Nilai default pelanggan
                                              step=0.01, 
                                              help="Harga maksimum tetap mid peak (max: 17.458221)"),
    'price_diff_off_peak_var': st.number_input('Price Diff Off Peak Var', 
                                               min_value=0.0, 
                                               max_value=float(0.236095 * 3),  # 3x lipat
                                               value=0.028554,  # Nilai default pelanggan
                                               step=0.001, 
                                               help="Perbedaan harga variabel off peak (max: 0.236095)"),
    'price_diff_mid_peak_var': st.number_input('Price Diff Mid Peak Var', 
                                               min_value=0.0, 
                                               max_value=float(0.114102 * 3),  # 3x lipat
                                               value=0.073873,  # Nilai default pelanggan
                                               step=0.001, 
                                               help="Perbedaan harga variabel mid peak (max: 0.114102)"),
    'day_since_date_end': st.number_input('Days Since Date End', 
                                          min_value=0.0, 
                                          max_value=float(4795 * 3),  # 3x lipat
                                          value=1096.0,  # Nilai default pelanggan
                                          help="Hari sejak tanggal berakhir (max: 4795)"),
    
    # Categorical Features
    'channel_sales_MISSING': st.selectbox('Channel Sales Missing', [0, 1], index=int(1)),
    'channel_sales_foosdfpfkusacimwkcsosbicdxkicaua': st.selectbox('Channel Sales Foo', [0, 1], index=int(1)),
    'channel_sales_lmkebamcaaclubfxadlmueccxoimlema': st.selectbox('Channel Sales LMKE', [0, 1], index=int(0)),
    'has_gas_f': st.selectbox('Has Gas (False)', [0, 1], index=int(0)),
    'has_gas_t': st.selectbox('Has Gas (True)', [0, 1], index=int(1)),
    'origin_up_kamkkxfxxuwbdslkwifmmcsiusiuosws': st.selectbox('Origin UP Kamkk', [0, 1], index=int(0)),
    'origin_up_ldkssxwpmemidmecebumciepifcamkci': st.selectbox('Origin UP LDKS', [0, 1], index=int(0)),
    'origin_up_lxidpiddsbxsbosboudacockeimpuepw': st.selectbox('Origin UP LXID', [0, 1], index=int(1))
}

# Convert Features to DataFrame
input_data = pd.DataFrame([features])

# Load Model
model = load_model()

# Dapatkan nama fitur yang dilihat saat pelatihan
train_features = model.feature_names_in_

# Periksa apakah ada fitur yang hilang
missing_features = [feature for feature in train_features if feature not in input_data.columns]

# Tambahkan fitur yang hilang dengan nilai default 0
for feature in missing_features:
    input_data[feature] = 0

# Pastikan urutan kolom sesuai dengan urutan pelatihan
input_data = input_data[train_features]

# Tampilkan fitur yang hilang jika ada
# if missing_features:
#     st.error(f"Fitur yang hilang dari input (ditambahkan secara otomatis): {missing_features}")
# else:
#     st.success("Semua fitur lengkap!")

# Predict Churn
# Function to generate strategy recommendations using Gemini
def generate_churn_strategy(churn_probability, input_data):
    # Prepare context for Gemini prompt
    # Complete context for Gemini prompt
    context = f"""
    Customer Churn Analysis:
    - Churn Probability: {churn_probability:.2f}%
    - Active Products: {input_data['nb_prod_act'].values[0]}
    - Customer Tenure: {input_data['num_years_antig'].values[0]} years
    - Net Margin: ${input_data['net_margin'].values[0]:.2f}
    - Total Consumption (12 months): {input_data['cons_12m'].values[0]:.2f} kWh
    - Gas Consumption (12 months): {input_data['cons_gas_12m'].values[0]:.2f} kWh
    - Consumption Last Month: {input_data['cons_last_month'].values[0]:.2f} kWh
    - Forecast Meter Rent (12 months): ${input_data['forecast_meter_rent_12m'].values[0]:.2f}
    - Forecast Energy Price Peak: ${input_data['forecast_price_energy_peak'].values[0]:.4f}
    - Gross Margin Power Elec: ${input_data['margin_gross_pow_ele'].values[0]:.2f}
    - Net Margin Power Elec: ${input_data['margin_net_pow_ele'].values[0]:.2f}
    - Max Power: {input_data['pow_max'].values[0]:.2f} kW
    - Average Price (Peak Variable): ${input_data['avg_price_peak_var'].values[0]:.4f}
    - Average Price (Mid Peak Variable): ${input_data['avg_price_mid_peak_var'].values[0]:.4f}
    - Average Price (Off Peak Fixed): ${input_data['avg_price_off_peak_fix'].values[0]:.2f}
    - Average Price (Peak Fixed): ${input_data['avg_price_peak_fix'].values[0]:.2f}
    - Average Price (Mid Peak Fixed): ${input_data['avg_price_mid_peak_fix'].values[0]:.2f}
    - Max Price (Peak Variable): ${input_data['max_price_peak_var'].values[0]:.4f}
    - Max Price (Mid Peak Variable): ${input_data['max_price_mid_peak_var'].values[0]:.4f}
    - Max Price (Off Peak Fixed): ${input_data['max_price_off_peak_fix'].values[0]:.2f}
    - Max Price (Peak Fixed): ${input_data['max_price_peak_fix'].values[0]:.2f}
    - Max Price (Mid Peak Fixed): ${input_data['max_price_mid_peak_fix'].values[0]:.2f}
    - Price Difference (Off Peak Variable): ${input_data['price_diff_off_peak_var'].values[0]:.4f}
    - Price Difference (Mid Peak Variable): ${input_data['price_diff_mid_peak_var'].values[0]:.4f}
    - Days Since Contract End: {input_data['day_since_date_end'].values[0]:.0f}

    Categorical Attributes:
    - Sales Channel Missing: {'Yes' if input_data['channel_sales_MISSING'].values[0] == 1 else 'No'}
    - Sales Channel (Foo): {'Yes' if input_data['channel_sales_foosdfpfkusacimwkcsosbicdxkicaua'].values[0] == 1 else 'No'}
    - Sales Channel (LMKE): {'Yes' if input_data['channel_sales_lmkebamcaaclubfxadlmueccxoimlema'].values[0] == 1 else 'No'}
    - Has Gas (True): {'Yes' if input_data['has_gas_t'].values[0] == 1 else 'No'}
    - Has Gas (False): {'Yes' if input_data['has_gas_f'].values[0] == 1 else 'No'}
    - Origin UP (Kamkk): {'Yes' if input_data['origin_up_kamkkxfxxuwbdslkwifmmcsiusiuosws'].values[0] == 1 else 'No'}
    - Origin UP (LDKS): {'Yes' if input_data['origin_up_ldkssxwpmemidmecebumciepifcamkci'].values[0] == 1 else 'No'}
    - Origin UP (LXID): {'Yes' if input_data['origin_up_lxidpiddsbxsbosboudacockeimpuepw'].values[0] == 1 else 'No'}
    """
    
    # Define Gemini prompt
    prompt = f"""Based on the following customer data and high churn risk, 
    provide personalized retention strategies to customer:

    {context}

    Develop a comprehensive strategy to reduce churn probability customer and Improve customer satisfaction :
    1. Based on quantify context provide detailed personalized offerings and specific actionable recommendations to attract retention ."""

    # Use Gemini Pro model
    model = genai.GenerativeModel('gemini-pro')
    
    # Generate recommendation
    response = model.generate_content(prompt)
    
    return response.text

# Modify your prediction section
if st.button('Predict'):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)
    churn_probability = prediction_proba[0][1] * 100
    churn_status = "Churn" if prediction[0] == 1 else "Not Churn"
    
    st.subheader(f"Prediction: {churn_status}")
    st.subheader(f"The probability that this customer will churn is: {churn_probability:.2f}%")

    # Generate Gemini Strategy Recommendations
    if churn_probability > 50:  # High churn risk threshold
        st.warning("🚨 High Churn Risk Detected!")
        st.subheader("Retention Strategies by Gemini")
        
        try:
            # Generate and display Gemini recommendations
            gemini_recommendations = generate_churn_strategy(churn_probability, input_data)
            st.markdown(gemini_recommendations)
        except Exception as e:
            st.error(f"Error generating recommendations: {e}")
    
    # Show Input Data
    # st.write("### Input Data:")
    # st.dataframe(input_data)
