import streamlit as st
import pandas as pd
import joblib
import os
import google.generativeai as genai
from dotenv import load_dotenv
import warnings

class ChurnPredictionApp:
    def __init__(self):
        load_dotenv()
        self.setup_environment()
        self.model = self.load_model()
        self.data_analysis = self.load_and_prepare_data()
        st.set_page_config(page_title="Customer Churn Prediction", page_icon="🔮", layout="wide")

    def setup_environment(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            st.error("GEMINI_API_KEY not found in environment variables.")
            st.stop()
        genai.configure(api_key=self.gemini_api_key)

    def load_model(self):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                return joblib.load("rf_classifier_model.pkl")
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.stop()

    def load_and_prepare_data(self):
        try:
            df = pd.read_csv("df_clean.csv")
            feature_ranges = {column: {
                "min": df[column].min(),
                "max": df[column].max(),
                "mean": df[column].mean(),
                "median": df[column].median(),
                "std": df[column].std(),
            } for column in df.select_dtypes(include=["float64", "int64"]).columns}

            model_features = self.model.feature_names_in_
            feature_importance = pd.DataFrame({"feature": model_features, "importance": self.model.feature_importances_}).sort_values("importance", ascending=False)

            return {
                "data": df,
                "feature_ranges": feature_ranges,
                "top_features": feature_importance.head(5)["feature"].tolist(),
            }
        except Exception as e:
            st.error(f"Failed to prepare data: {e}")
            st.stop()

    def generate_general_insights(self):
        try:
            model = genai.GenerativeModel("gemini-pro")
            prompt = f"Generate a comprehensive analysis with:
            1. Main patterns in the data
            2. Key factors influencing churn
            3. Strategic recommendations
            4. In-depth business insights
            on the dataset with {len(self.data_analysis['data'])} customers and a churn rate of {self.data_analysis['data']['churn'].mean()*100:.2f}%. Top features: {', '.join(self.data_analysis['top_features'])}."
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating insights: {str(e)}"

    def generate_personalized_recommendations(self, features, churn_probability):
        try:
            model = genai.GenerativeModel("gemini-pro")
            user_context = "\n".join([f"{k}: {v}" for k, v in features.items()])
            prompt = f"Churn Probability: {churn_probability:.2f}%. Customer Context:\n{user_context}. Generate:
            1. Detailed risk assessment
            2. Personalized retention strategies
            3. Targeted intervention recommendations
            4. Specific actions to reduce churn risk
            5. Personalized customer engagement approaches."
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating recommendations: {str(e)}"

    def create_feature_inputs(self):
        features = {
            'cons_12m': st.number_input('Consumption 12m', min_value=0.0, max_value=float(6_207_104 * 3), value=54946.0, help="Consumption in the last 12 months (max: 6,207,104)"),
            'cons_gas_12m': st.number_input('Gas Consumption 12m', min_value=0.0, max_value=float(4_154_590 * 3), value=0.0, help="Gas consumption in the last 12 months (max: 4,154,590)"),
            'cons_last_month': st.number_input('Consumption Last Month', min_value=0.0, max_value=float(771_203 * 3), value=0.0, help="Consumption last month (max: 771,203)"),
            'forecast_meter_rent_12m': st.number_input('Forecast Meter Rent 12m', min_value=0.0, max_value=float(599.31 * 3), value=1.78, step=0.01, help="Forecast meter rent for 12 months (max: 599.31)"),
            'forecast_price_energy_peak': st.number_input('Forecast Energy Price Peak', min_value=0.0, max_value=float(0.195975 * 3), value=0.098142, step=0.001, help="Forecast peak energy price (max: 0.195975)"),
            'margin_gross_pow_ele': st.number_input('Gross Margin Power Ele', min_value=0.0, max_value=float(374.64 * 3), value=25.44, step=0.01, help="Gross margin for power electricity (max: 374.64)"),
            'margin_net_pow_ele': st.number_input('Net Margin Power Ele', min_value=0.0, max_value=float(374.64 * 3), value=25.44, step=0.01, help="Net margin for power electricity (max: 374.64)"),
            'nb_prod_act': st.number_input('Number of Active Products', min_value=1.0, max_value=float(32 * 3), value=2.0, help="Number of active products (max: 32)"),
            'net_margin': st.number_input('Net Margin', min_value=0.0, max_value=float(24_570 * 3), value=678.99, help="Net margin (max: 24,570)"),
            'num_years_antig': st.number_input('Number of Years', min_value=1.0, max_value=float(13 * 3), value=3.0, help="Number of years (max: 13)"),
            'pow_max': st.number_input('Max Power', min_value=3.3, max_value=float(320.0 * 3), value=43.648, step=0.1, help="Maximum power (max: 320.0)"),
            'avg_price_peak_var': st.number_input('Avg Price Peak Var', min_value=0.0, max_value=float(0.196 * 3), value=0.100749, step=0.001, help="Average variable peak price (max: 0.196)"),
            'avg_price_mid_peak_var': st.number_input('Avg Price Mid Peak Var', min_value=0.0, max_value=float(0.102951 * 3), value=0.066530, step=0.001, help="Average variable mid peak price (max: 0.102951)"),
            'avg_price_off_peak_fix': st.number_input('Avg Price Off Peak Fix', min_value=0.0, max_value=float(59.286 * 3), value=40.942265, step=0.01, help="Average fixed off peak price (max: 59.286)"),
            'avg_price_peak_fix': st.number_input('Avg Price Peak Fix', min_value=0.0, max_value=float(36.490689 * 3), value=22.352010, step=0.01, help="Average fixed peak price (max: 36.490 689)"),
            'avg_price_mid_peak_fix': st.number_input('Avg Price Mid Peak Fix', min_value=0.0, max_value=float(16.818917 * 3), value=14.901340, step=0.01, help="Average fixed mid peak price (max: 16.818917)"),
            'max_price_peak_var': st.number_input('Max Price Peak Var', min_value=0.0, max_value=float(0.229788 * 3), value=0.103963, step=0.001, help="Maximum variable peak price (max: 0.229788)"),
            'max_price_mid_peak_var': st.number_input('Max Price Mid Peak Var', min_value=0.0, max_value=float(0.114102 * 3), value=0.073873, step=0.001, help="Maximum variable mid peak price (max: 0.114102)"),
            'max_price_off_peak_fix': st.number_input('Max Price Off Peak Fix', min_value=0.0, max_value=float(59.444710 * 3), value=44.266930, step=0.01, help="Maximum fixed off peak price (max: 59.444710)"),
            'max_price_peak_fix': st.number_input('Max Price Peak Fix', min_value=0.0, max_value=float(36.490689 * 3), value=24.43733, step=0.01, help="Maximum fixed peak price (max: 36.490689)"),
            'max_price_mid_peak_fix': st.number_input('Max Price Mid Peak Fix', min_value=0.0, max_value=float(17.458221 * 3), value=16.291555, step=0.01, help="Maximum fixed mid peak price (max: 17.458221)"),
            'price_diff_off_peak_var': st.number_input('Price Diff Off Peak Var', min_value=0.0, max_value=float(0.236095 * 3), value=0.028554, step=0.001, help="Difference in variable off peak price (max: 0.236095)"),
            'price_diff_mid_peak_var': st.number_input('Price Diff Mid Peak Var', min_value=0.0, max_value=float(0.114102 * 3), value=0.073873, step=0.001, help="Difference in variable mid peak price (max: 0.114102)"),
            'day_since_date_end': st.number_input('Days Since Date End', min_value=731.0, max_value=float(4795 * 3), value=1096.0, help="Days since end date (max: 4795)"),
            'channel_sales_MISSING': st.selectbox('Channel Sales Missing', [0, 1], index=int(1)),
            'channel_sales_foosdfpfkusacimwkcsosbicdxkicaua': st.selectbox('Channel Sales Foo', [0, 1], index=int(1)),
            'channel_sales_lmkebamcaaclubfxadlmueccxoimlema': st.selectbox('Channel Sales LMKE', [0, 1], index=int(0)),
            'has_gas_f': st.selectbox('Has Gas (False)', [0, 1], index=int(0)),
            'has_gas_t': st.selectbox('Has Gas (True)', [0, 1], index=int(1)),
            'origin_up_kamkkxfxxuwbdslkwifmmcsiusiuosws': st.selectbox('Origin UP Kamkk', [0, 1], index=int(0)),
            'origin_up_ldkssxwpmemidmecebumciepifcamkci': st.selectbox('Origin UP LDKS', [0, 1], index=int(0)),
            'origin_up_lxidpiddsbxsbosboudacockeimpuepw': st.selectbox('Origin UP LXID', [0, 1], index=int(1))
        }
        return features

    def run(self):
        st.title("🔮 Customer Churn Prediction and Retention Strategies")
        tab1, tab2 = st.tabs(["General Insights", "Personal Prediction"])

        with tab1:
            st.header("General Insights & Strategy for Retention")
            general_insights = self.generate_general_insights()
            st.markdown(general_insights)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Customers", len(self.data_analysis['data']))
            with col2:
                st.metric("Churn Rate", f"{self.data_analysis['data']['churn'].mean()*100:.2f}%")

        with tab2:
            st.header("Personal Churn Prediction")
            features = self.create_feature_inputs()

            if st.button('Predict Churn Risk'):
                try:
                    input_df = pd.DataFrame([features])
                    prediction_proba = self.model.predict_proba(input_df)
                    churn_probability = prediction_proba[0][1] * 100

                    st.metric("Churn Probability", f"{churn_probability:.2f}%")

                    if churn_probability > 50:
                        st.warning("🚨 High Churn Risk Detected!")

                    personalized_recommendations = self.generate_personalized_recommendations(features, churn_probability)
                    st.markdown(personalized_recommendations)

                except Exception as e:
                    st.error(f"Error in prediction: {str(e)}")

def main():
    app = ChurnPredictionApp()
    app.run()

if __name__ == "__main__":
    main()
