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

# Semua fitur sebagai input utama (pastikan semua angka dalam tipe float)
features = {
    'cons_12m': st.number_input('Consumption 12m', 
                                min_value=0.0, 
                                max_value=float(6_207_104 * 3),  # 3x lipat
                                value=124671.0, 
                                help="Konsumsi dalam 12 bulan terakhir (max: 6,207,104)"),
    'cons_gas_12m': st.number_input('Gas Consumption 12m', 
                                    min_value=0.0, 
                                    max_value=float(4_154_590 * 3),  # 3x lipat
                                    value=19880.0, 
                                    help="Konsumsi gas dalam 12 bulan terakhir (max: 4,154,590)"),
    'cons_last_month': st.number_input('Consumption Last Month', 
                                       min_value=0.0, 
                                       max_value=float(771_203 * 3),  # 3x lipat
                                       value=12310.0, 
                                       help="Konsumsi bulan terakhir (max: 771,203)"),
    'forecast_meter_rent_12m': st.number_input('Forecast Meter Rent 12m', 
                                               min_value=0.0, 
                                               max_value=float(599.31 * 3),  # 3x lipat
                                               value=66.88, 
                                               step=0.01, 
                                               help="Perkiraan sewa meter 12 bulan (max: 599.31)"),
    'forecast_price_energy_peak': st.number_input('Forecast Energy Price Peak', 
                                                  min_value=0.0, 
                                                  max_value=float(0.195975 * 3),  # 3x lipat
                                                  value=0.052, 
                                                  step=0.001, 
                                                  help="Perkiraan harga energi puncak (max: 0.195975)"),
    'margin_gross_pow_ele': st.number_input('Gross Margin Power Ele', 
                                            min_value=0.0, 
                                            max_value=float(374.64 * 3),  # 3x lipat
                                            value=27.05, 
                                            step=0.01, 
                                            help="Margin kotor daya listrik (max: 374.64)"),
    'margin_net_pow_ele': st.number_input('Net Margin Power Ele', 
                                          min_value=0.0, 
                                          max_value=float(374.64 * 3),  # 3x lipat
                                          value=27.05, 
                                          step=0.01, 
                                          help="Margin bersih daya listrik (max: 374.64)"),
    'nb_prod_act': st.number_input('Number of Active Products', 
                                   min_value=1.0,  # Pastikan float
                                   max_value=float(32 * 3),  # 3x lipat
                                   value=1.0, 
                                   help="Jumlah produk aktif (max: 32)"),
    'net_margin': st.number_input('Net Margin', 
                                  min_value=0.0, 
                                  max_value=float(24_570 * 3),  # 3x lipat
                                  value=206.0, 
                                  help="Margin bersih (max: 24,570)"),
    'num_years_antig': st.number_input('Number of Years', 
                                       min_value=1.0,  # Pastikan float
                                       max_value=float(13 * 3),  # 3x lipat
                                       value=4.0, 
                                       help="Jumlah tahun (max: 13)"),
    'pow_max': st.number_input('Max Power', 
                               min_value=3.3,  # Pastikan float
                               max_value=float(320.0 * 3),  # 3x lipat
                               value=18.7, 
                               step=0.1, 
                               help="Daya maksimum (max: 320.0)"),
    'avg_price_peak_var': st.number_input('Avg Price Peak Var', 
                                          min_value=0.0, 
                                          max_value=float(0.196 * 3),  # 3x lipat
                                          value=0.053, 
                                          step=0.001, 
                                          help="Rata-rata harga variabel puncak (max: 0.196)"),
    'avg_price_mid_peak_var': st.number_input('Avg Price Mid Peak Var', 
                                              min_value=0.0, 
                                              max_value=float(0.102951 * 3),  # 3x lipat
                                              value=0.030, 
                                              step=0.001, 
                                              help="Rata-rata harga variabel mid peak (max: 0.102951)"),
    'avg_price_off_peak_fix': st.number_input('Avg Price Off Peak Fix', 
                                              min_value=0.0, 
                                              max_value=float(59.286 * 3),  # 3x lipat
                                              value=43.041, 
                                              step=0.01, 
                                              help="Rata-rata harga tetap off peak (max: 59.286)"),
    'avg_price_peak_fix': st.number_input('Avg Price Peak Fix', 
                                          min_value=0.0, 
                                          max_value=float(36.490689 * 3),  # 3x lipat
                                          value=0.0, 
                                          step=0.01, 
                                          help="Rata-rata harga tetap puncak (max: 36.490689)"),
    'avg_price_mid_peak_fix': st.number_input('Avg Price Mid Peak Fix', 
                                              min_value=0.0, 
                                              max_value=float(16.818917 * 3),  # 3x lipat
                                              value=0.0, 
                                              step=0.01, 
                                              help="Rata-rata harga tetap mid peak (max: 16.818917)"),
    'max_price_peak_var': st.number_input('Max Price Peak Var', 
                                          min_value=0.0, 
                                          max_value=float(0.229788 * 3),  # 3x lipat
                                          value=0.0, 
                                          step=0.001, 
                                          help="Harga maksimum variabel puncak (max: 0.229788)"),
    'max_price_mid_peak_var': st.number_input('Max Price Mid Peak Var', 
                                              min_value=0.0, 
                                              max_value=float(0.114102 * 3),  # 3x lipat
                                              value=0.0, 
                                              step=0.001, 
                                              help="Harga maksimum variabel mid peak (max: 0.114102)"),
    'max_price_off_peak_fix': st.number_input('Max Price Off Peak Fix', 
                                              min_value=0.0, 
                                              max_value=float(59.444710 * 3),  # 3x lipat
                                              value=0.0, 
                                              step=0.01, 
                                              help="Harga maksimum tetap off peak (max: 59.444710)"),
    'max_price_peak_fix': st.number_input('Max Price Peak Fix', 
                                          min_value=0.0, 
                                          max_value=float(36.490689 * 3),  # 3x lipat
                                          value=0.0, 
                                          step=0.01, 
                                          help="Harga maksimum tetap puncak (max: 36.490689)"),
    'max_price_mid_peak_fix': st.number_input('Max Price Mid Peak Fix', 
                                              min_value=0.0, 
                                              max_value=float(17.458221 * 3),  # 3x lipat
                                              value=0.0, 
                                              step=0.01, 
                                              help="Harga maksimum tetap mid peak (max: 17.458221)"),
    'price_diff_off_peak_var': st.number_input('Price Diff Off Peak Var', 
                                               min_value=0.0, 
                                               max_value=float(0.236095 * 3),  # 3x lipat
                                               value=0.01, 
                                               step=0.001, 
                                               help="Perbedaan harga variabel off peak (max: 0.236095)"),
    'price_diff_mid_peak_var': st.number_input('Price Diff Mid Peak Var', 
                                               min_value=0.0, 
                                               max_value=float(0.114102 * 3),  # 3x lipat
                                               value=0.0, 
                                               step=0.001, 
                                               help="Perbedaan harga variabel mid peak (max: 0.114102)"),
    'day_since_date_end': st.number_input('Days Since Date End', 
                                          min_value=731.0, 
                                          max_value=float(4795 * 3),  # 3x lipat
                                          value=1946.0, 
                                          help="Hari sejak tanggal berakhir (max: 4795)"),
    
    # Categorical Features
    'channel_sales_MISSING': st.selectbox('Channel Sales Missing', [0, 1]),
    'channel_sales_foosdfpfkusacimwkcsosbicdxkicaua': st.selectbox('Channel Sales Foo', [0, 1]),
    'channel_sales_lmkebamcaaclubfxadlmueccxoimlema': st.selectbox('Channel Sales LMKE', [0, 1]),
    'has_gas_f': st.selectbox('Has Gas (False)', [0, 1]),
    'has_gas_t': st.selectbox('Has Gas (True)', [0, 1]),
    'origin_up_kamkkxfxxuwbdslkwifmmcsiusiuosws': st.selectbox('Origin UP Kamkk', [0, 1]),
    'origin_up_ldkssxwpmemidmecebumciepifcamkci': st.selectbox('Origin UP LDKS', [0, 1]),
    'origin_up_lxidpiddsbxsbosboudacockeimpuepw': st.selectbox('Origin UP LXID', [0, 1])
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
if missing_features:
    st.error(f"Fitur yang hilang dari input (ditambahkan secara otomatis): {missing_features}")
else:
    st.success("Semua fitur lengkap!")

# Predict Churn
if st.button('Predict'):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)
    churn_status = "Churn" if prediction[0] == 1 else "Not Churn"
    st.subheader(f"Prediction: {churn_status}")
    st.subheader(f"The probability that this customer will churn is: {prediction_proba[0][1]*100:.2f}%")

    # Show Input Data
    st.write("### Input Data:")
    st.dataframe(input_data)