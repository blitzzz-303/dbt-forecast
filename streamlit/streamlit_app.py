import streamlit as st, pandas as pd
from utils import gpt_utils
import streamlit_app_viz as v

st.set_page_config(
    page_title="Ex-stream-ly Cool App",
    page_icon=":smiling_imp:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)


user_prompt = st.text_input('Write your database inquiry in natural language')

if user_prompt.strip() == '':
    exit()
gpt = gpt_utils()

@st.cache_data
def _classify_prompt(user_prompt):
    return gpt.classify_prompt(user_prompt)
df, gpt_response, prompt_summary, user_prompt_hash = _classify_prompt(user_prompt)



if df is None:
    st.error('Failed to get the result from Snowflake')
    st.info('GPT-3 response: ', gpt_response)
    # exit the program
    exit()
else:
    st.info(f'Target: {prompt_summary}')
    st.code(f'-- generated SQL query {gpt_response}')
    st.write(df)

st.sidebar.title("Filters")
# convert the week_date from string to date time
df["WEEK_DATE"] = pd.to_datetime(df["WEEK_DATE"])
start_date = st.sidebar.date_input("Select start date", min(df["WEEK_DATE"]))
end_date = st.sidebar.date_input("Select end date", max(df["WEEK_DATE"]))
# selected_store = st.sidebar.selectbox("Select store", df["STORE_DEPT_PK"].unique())

# create multiple button grid for each store dept pk in the side bar
list_of_store_dept_pk = df["STORE_DEPT_PK"].unique()
list_of_store_dept_pk.sort()
# add to a grid of radio buttons
selected_store = st.sidebar.radio("Select store", list_of_store_dept_pk)

# Filter data based on user selection
# convert start_date and end_date to datetime64[ns]
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)
filtered_df = df[(df["WEEK_DATE"] >= start_date)
                 & (df["WEEK_DATE"] <= end_date)
                 & (df["STORE_DEPT_PK"] == selected_store)]

pred = v.generate_predict_viz(filtered_df)
st.plotly_chart(pred, use_container_width=True)

h = v.generate_holiday_chart(filtered_df)
st.plotly_chart(h, use_container_width=True)

p = v.generate_temp_line_chart(filtered_df)
st.plotly_chart(p, use_container_width=True)

st.balloons()