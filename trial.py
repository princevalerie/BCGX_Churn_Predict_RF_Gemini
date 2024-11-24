import streamlit as st
import pandas as pd
import joblib

# Load Model
@st.cache_resource
def load_model():
    return joblib.load('rf_classifier_model.pkl')

# Streamlit App Title
st.title("Customer Churn Prediction App")

# Input Features
st.header("Input Features")

# Input nilai besar (max_value = 3 kali lipat)
features = {
    'cons_12m': st.number_input('Consumption 12m', 
                                min_value=0, 
                                max_value=6_207_104 * 3,  # 3x lipat
                                value=124671, 
                                help="Konsumsi dalam 12 bulan terakhir (max: 6,207,104)"),
    'cons_gas_12m': st.number_input('Gas Consumption 12m', 
                                    min_value=0, 
                                    max_value=4_154_590 * 3,  # 3x lipat
                                    value=19880, 
                                    help="Konsumsi gas dalam 12 bulan terakhir (max: 4,154,590)"),
    'cons_last_month': st.number_input('Consumption Last Month', 
                                       min_value=0, 
                                       max_value=771_203 * 3,  # 3x lipat
                                       value=12310, 
                                       help="Konsumsi bulan terakhir (max: 771,203)"),
    'net_margin': st.number_input('Net Margin', 
                                  min_value=0, 
                                  max_value=24_570 * 3,  # 3x lipat
                                  value=206, 
                                  help="Margin bersih (max: 24,570)"),
    'day_since_date_end': st.number_input('Days Since Date End', 
                                          min_value=731, 
                                          max_value=4795 * 3,  # 3x lipat
                                          value=1946, 
                                          help="Hari sejak tanggal berakhir (max: 4795)"),
}

# Input lainnya (max_value = 2 kali lipat untuk sidebar)
st.sidebar.header("Other Features")
features.update({
    'forecast_meter_rent_12m': st.sidebar.number_input('Forecast Meter Rent 12m', 
                                                       min_value=0.0, 
                                                       max_value=599.31 * 2,  # 2x lipat
                                                       value=66.88, 
                                                       step=0.01, 
                                                       help="Perkiraan sewa meter 12 bulan (max: 599.31)"),
    'forecast_price_energy_peak': st.sidebar.number_input('Forecast Energy Price Peak', 
                                                          min_value=0.0, 
                                                          max_value=0.195975 * 2,  # 2x lipat
                                                          value=0.052, 
                                                          step=0.001, 
                                                          help="Perkiraan harga energi puncak (max: 0.195975)"),
    'margin_gross_pow_ele': st.sidebar.number_input('Gross Margin Power Ele', 
                                                    min_value=0.0, 
                                                    max_value=374.64 * 2,  # 2x lipat
                                                    value=27.05, 
                                                    step=0.01, 
                                                    help="Margin kotor daya listrik (max: 374.64)"),
    'margin_net_pow_ele': st.sidebar.number_input('Net Margin Power Ele', 
                                                  min_value=0.0, 
                                                  max_value=374.64 * 2,  # 2x lipat
                                                  value=27.05, 
                                                  step=0.01, 
                                                  help="Margin bersih daya listrik (max: 374.64)"),
    'nb_prod_act': st.sidebar.number_input('Number of Active Products', 
                                           min_value=1, 
                                           max_value=32 * 2,  # 2x lipat
                                           value=1, 
                                           help="Jumlah produk aktif (max: 32)"),
    'num_years_antig': st.sidebar.number_input('Number of Years', 
                                               min_value=1, 
                                               max_value=13 * 2,  # 2x lipat
                                               value=4, 
                                               help="Jumlah tahun (max: 13)"),
    'pow_max': st.sidebar.number_input('Max Power', 
                                       min_value=3.3, 
                                       max_value=320.0 * 2,  # 2x lipat
                                       value=18.7, 
                                       step=0.1, 
                                       help="Daya maksimum (max: 320.0)"),
    'avg_price_peak_var': st.sidebar.number_input('Avg Price Peak Var', 
                                                  min_value=0.0, 
                                                  max_value=0.196 * 2,  # 2x lipat
                                                  value=0.053, 
                                                  step=0.001, 
                                                  help="Rata-rata harga variabel puncak (max: 0.196)"),
    'avg_price_mid_peak_var': st.sidebar.number_input('Avg Price Mid Peak Var', 
                                                      min_value=0.0, 
                                                      max_value=0.102951 * 2,  # 2x lipat
                                                      value=0.030, 
                                                      step=0.001, 
                                                      help="Rata-rata harga variabel mid peak (max: 0.102951)"),
    'avg_price_off_peak_fix': st.sidebar.number_input('Avg Price Off Peak Fix', 
                                                      min_value=0.0, 
                                                      max_value=59.286 * 2,  # 2x lipat
                                                      value=43.041, 
                                                      step=0.01, 
                                                      help="Rata-rata harga tetap off peak (max: 59.286)"),
    'price_diff_off_peak_var': st.sidebar.number_input('Price Diff Off Peak Var', 
                                                       min_value=0.0, 
                                                       max_value=0.236 * 2,  # 2x lipat
                                                       value=0.01, 
                                                       step=0.001, 
                                                       help="Perbedaan harga variabel off peak (max: 0.236)"),
    # Sidebar selection tetap tidak berubah
    'channel_sales_MISSING': st.sidebar.selectbox('Channel Sales Missing', [0, 1]),
    'channel_sales_foosdfpfkusacimwkcsosbicdxkicaua': st.sidebar.selectbox('Channel Sales Foo', [0, 1]),
    'channel_sales_lmkebamcaaclubfxadlmueccxoimlema': st.sidebar.selectbox('Channel Sales LMKE', [0, 1]),
    'has_gas_f': st.sidebar.selectbox('Has Gas (False)', [0, 1]),
    'has_gas_t': st.sidebar.selectbox('Has Gas (True)', [0, 1]),
    'origin_up_kamkkxfxxuwbdslkwifmmcsiusiuosws': st.sidebar.selectbox('Origin UP Kamkk', [0, 1]),
    'origin_up_ldkssxwpmemidmecebumciepifcamkci': st.sidebar.selectbox('Origin UP LDKS', [0, 1]),
    'origin_up_lxidpiddsbxsbosboudacockeimpuepw': st.sidebar.selectbox('Origin UP LXID', [0, 1]),
})

# Convert Features to DataFrame
input_data = pd.DataFrame([features])

# Load Model
model = load_model()

# Predict Churn
if st.button('Predict'):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)
    churn_status = "Churn" if prediction[0] == 1 else "Not Churn"
    st.subheader(f"Prediction: {churn_status}")
    st.subheader(f"Prediction Probability: {prediction_proba[0][1]*100:.2f}%")

    # Show Input Data
    st.write("### Input Data:")
    st.dataframe(input_data)