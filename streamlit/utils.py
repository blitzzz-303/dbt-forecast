from scipy.spatial.distance import cdist
import streamlit as st, os, time, hashlib, ruamel.yaml as yaml, logging, importlib
import tensorflow_hub as hub, tensorflow as tf, numpy as np
from snowflake.snowpark import Session
import vertexai
import openai
from vertexai.preview.language_models import TextGenerationModel, TextGenerationResponse

# openai.api_key = os.getenv("OPENAI_API_KEY")
current_dir = os.path.dirname(os.path.realpath(__file__))
logging.basicConfig(level=logging.INFO)

vertexai.init(project=os.getenv('GCP_PROJECT_ID'), location=os.getenv('GCP_LOCATION'))
class gpt_utils():
    session = None
    def __init__(self, **kwargs):
        self.create_snowflake_connection()
        self.__dict__.update(kwargs)

    def gpt_log(self, message, level='info'):
        if level == 'info':
            logging.info(f'INFO: {message}')
        elif level == 'error':
            logging.error(f'ERROR: {message}')
        elif level == 'warning':
            logging.warning(f'WARNING: {message}')
        elif level == 'debug':
            logging.debug(f'DEBUG: {message}')

    def gpt_adapter(self, **kwargs):
        # temporary disable vertex
        return self.gpt_request(**kwargs)

        if kwargs.get('provider', 'vertex') == 'vertex':
            return self.vertex_gpt(kwargs['prompt'])
        else:
            return self.gpt_request(**kwargs)

    def vertex_gpt(self, prompt: str, model_name='text-bison@001', temperature=0,
                        max_decode_steps=1024, top_p=0.5, top_k=40) :
        model = TextGenerationModel.from_pretrained(model_name)
        response = model.predict(prompt, temperature=temperature,
                                    max_output_tokens=max_decode_steps, top_k=top_k, top_p=top_p)
        self.gpt_log(f"Response from Model: {response}")
        return response.text

    def gpt_request(self, **kwargs):
        # generate helloworld with chatGPT turbo 3.5
        response = openai.Completion.create(model=kwargs.get('model', 'text-davinci-003'),
                                            prompt=kwargs['prompt'],
                                            best_of=kwargs.get('best_of', 1),
                                            temperature=kwargs.get('temperature', 0),
                                            max_tokens=kwargs.get('max_tokens', 2000),
                                            top_p=kwargs.get('top_p', 1),
                                            frequency_penalty=kwargs.get('frequency_penalty', 0),
                                            presence_penalty=kwargs.get('presence_penalty', 0))
        # get the response from openAI
        self.gpt_log(f"Response from Model: {response.choices[0].text}")
        return response.choices[0].text
    
    def extract_code_from_response(self, template, response, key='######'):
        # split the response by ###### and get the 2nd part
        code = template + response
        logging.info(f'Code: {code}')
        code = code.split(key)[1]
        return code

    def get_dataframe(self, sys_prompt, shots=1):
        gpt_response, i = '', 0
        while i < 3:
            try: 
                gpt_response = self.gpt_adapter(**{'prompt': sys_prompt})
                gpt_response = self.extract_code_from_response(sys_prompt, gpt_response)
                # execute the gpt_response in Snowflake
                r = self.session.sql(gpt_response)
                # get the df from Snowflake
                df = r.to_pandas().copy()
                # if df has zero row, throw exception
                if df.shape[0] == 0:
                    raise Exception('No result from Snowflake')
                return df, gpt_response
            except Exception as e:
                time.sleep(3)
                i += 1
        return None, gpt_response

    def create_snowflake_connection(self):
        connection_parameters = {
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "user": os.getenv("SNOWFLAKE_USER"),
        "password": os.getenv("SNOWFLAKE_PASSWORD"),
        "role": os.getenv("SNOWFLAKE_ROLE"),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
        "database": os.getenv("SNOWFLAKE_DATABASE"),
        "schema": os.getenv("SNOWFLAKE_SCHEMA")}
        self.connection_parameters = connection_parameters
        self.session = Session.builder.configs(connection_parameters).create()

    def embed_query(self, query):
        if not hasattr(self, 'embedding_model'):
            model_dir = "./embedding_model"
            # get current directory
            current_dir = os.path.dirname(os.path.realpath(__file__))
            # get the model directory
            model_dir = os.path.join(current_dir, model_dir)
            # Download the USE model as a binary file
            if not os.path.exists(model_dir):
                print('Model not found, downloading process may take a while...')
                os.makedirs(model_dir)
                module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5?tf-hub-format=compressed"
                tf.saved_model.save(hub.load(module_url), model_dir)
            if not hasattr(self, 'embedding_model'):
                self.embedding_model = tf.saved_model.load(model_dir)
        query_embedding = self.embedding_model([query])
        return query_embedding


    def get_previous_advice(self, programing_language, gpt_summary_embedding):
        # check if the prompt is already in the database
        similar_prompt = self.get_similar_prompt(programing_language, gpt_summary_embedding)
        # self.gpt_log(f'Similar prompt: {similar_prompt}')
        # if similar_prompt:
        #     prompt_advice = '---\n\n'
        #     prompt_advice += f'Here is an example answer, with cosine_similarity {similar_prompt["similarity"][0]}, follow and updates SQL necessay components:\n\n'
        #     prompt_advice += f'{similar_prompt["gpt_response"]}\n\n'
        #     if similar_prompt.get('user_advices', []) != []:
        #         prompt_advice += f'--user advice(s):\n'
        #         for user_advice in similar_prompt['user_advices']:
        #             prompt_advice += f'---{user_advice}\n'
        #         prompt_advice += '\n'
        #     elif similar_prompt.get('inherit_advices', []) != []:
        #          prompt_advice += f'--user advice(s) (ALWAYS TOP PRIORITY): {similar_prompt["inherit_advice"]}\n\n'
        #     prompt_advice += '---\n\n'
        # else:
        #     prompt_advice = ''
        #     similar_prompt = {}
        return '', similar_prompt

    def classify_prompt(self, user_prompt):
        # get the prompt from user
        user_prompt = user_prompt.strip()
        # hash the prompt with sha256 and get first 10 characters
        user_prompt_hash = hashlib.sha256(user_prompt.encode()).hexdigest()[:10]
        s_template = {'prompt': gpt_utils.get_gpt_prompt(5).replace('##user_prompt##', user_prompt),
                      'provider': 'vertex'}
        prompt_summary = self.gpt_adapter(**s_template)
        prompt_summary = self.extract_code_from_response(s_template['prompt'], prompt_summary)
        # embed the summary
        gpt_summary_embedding = self.embed_query(prompt_summary)


        # get the programing language
        p_template = gpt_utils.get_gpt_prompt(4).replace('##user_prompt##', user_prompt)
        # programing_language = self.gpt_adapter(**{'prompt': p_template, 'provider': 'vertex'}).strip().lower()
        programing_language = 'sql'

        self.gpt_log(f'Programing language: {programing_language}')
        # prompt_advice, similar_prompt = self.get_previous_advice(programing_language, gpt_summary_embedding)
        prompt_advice, similar_prompt = '', {}

        # chec if programing_language is sql
        if 'sql' in programing_language:
            df, gpt_response = self.get_df_from_sql_prompt(user_prompt, prompt_advice)
            # if df is None, return the error message
            # self.store_prompt(user_prompt, programing_language, user_prompt_hash,
            #                     prompt_summary, gpt_summary_embedding, gpt_response,
            #                     similar_prompt.get('inherit_advices', []), similar_prompt.get('user_advices', []))
            if df is None or df.shape[0] == 0:
                return None, f'Error: We currently only support SQL, please try again later', None, None
            return df, gpt_response, prompt_summary, user_prompt_hash
        else:
            return None, f'Error: We currently only support SQL, please try again later', None, None

    def visualize_with_python(self, df):
        python_viz_excample, file_path = self.load_python_viz_example()
        prompt = gpt_utils.get_gpt_prompt(1.1)
        prompt = prompt.replace('##prompt_advice##', python_viz_excample)
        prompt = prompt.replace('##df_head##', str(df.head(3).to_dict()))
        prompt = prompt.replace('##df_dtype##', str(df.dtypes.to_dict()))
        i = '1'
        prompt = prompt.replace('##i##', i)
        gpt_response = self.gpt_adapter(**{'prompt': prompt})
        python_code = self.extract_code_from_response(prompt, gpt_response)
        # write the code to a file
        file_path = os.path.join(current_dir, f'gpt_responses/dynamoc_response_{i}.py')
        with open(file_path, 'w') as f:
            f.write(python_code)
        # import the function, with different method than exec()
        spec = importlib.util.spec_from_file_location("dynamic_response", file_path)
        dynamic_response = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dynamic_response)
        # call the function, the function name is streamlit_viz_{i}
        streamlit_viz = getattr(dynamic_response, f"streamlit_viz_{i}")
        fig = streamlit_viz(df)
        return fig


    def get_df_from_sql_prompt(self, user_prompt, prompt_advice, gpt_prompt=0.1):
        prompt_template = gpt_utils.get_gpt_prompt(0.1)
        # get the prompt from user
        prompt_template = prompt_template.replace('##user_prompt##', user_prompt)
        prompt_template = prompt_template.replace('##prompt_advice##', prompt_advice)
        # ask GPT to generate the query
        
        gpt_response = self.gpt_adapter(**{'prompt': prompt_template})
        gpt_response = self.extract_code_from_response(prompt_template, gpt_response)
        # get the dataframe from Snowflake
        df = self.get_snowflake_df_from_qry(gpt_response)
        return df, gpt_response

    def get_snowflake_df_from_qry(self, qry):
        try:
            df = self.session.sql(qry).to_pandas().copy()
        except Exception as e:
            # log the query
            self.gpt_log(f'Error from query: {qry}', 'error')
            raise Exception('No result from Snowflake - Error: {}'.format(e))
        return df

    def get_similar_prompt(self, programing_language, gpt_summary_embedding, threshold=0.7):
        prompt_info = {}
        score_list, user_advices, inherit_advices = [], [], []
        # get current directory
        current_dir = os.path.dirname(os.path.realpath(__file__))
        # get the model directory
        self.prompts_resource = os.path.join(current_dir, 'programing_language', programing_language)
        
        if not os.path.exists(self.prompts_resource):
            os.makedirs(self.prompts_resource)
        if not os.path.exists(f'{self.prompts_resource}/prompt_info.yaml'):
            with open(f'{self.prompts_resource}/prompt_info.yaml', 'w') as f:
                yaml.dump({'prompts': []}, f)
        # load the prompt_info.yaml
        
        with open(f'{self.prompts_resource}/prompt_info.yaml', 'r') as f:
            prompt_info = yaml.load(f, Loader=yaml.RoundTripLoader)
        
        # handle TypeError: 'NoneType' object is not iterable
        if prompt_info is None or prompt_info['prompts'] is None:
            return None
        # try:
        for saved_prompt in prompt_info['prompts']:
            vector = saved_prompt['vector']
            vector = np.array(vector)
            vector = tf.convert_to_tensor(vector)
            similarity_score = 1. - cdist(vector, gpt_summary_embedding, 'cosine')
            score_list.append({'prompt_summary': saved_prompt['content']['prompt_summary'],
                            'inherit_advices': saved_prompt['content'].get('inherit_advices', []),
                            'user_advices': saved_prompt['content'].get('user_advices', []),
                            'gpt_response': saved_prompt['content']['gpt_response'],
                            'similarity': similarity_score})

        if len(score_list) == 0:
            return None
        # filter score_list with non empty user_advice
        score_list_filter = [s for s in score_list if len(s['user_advices']) > 0]
        if len(score_list_filter) > 0:
            score_list_filter = sorted(score_list_filter, key=lambda x: x['similarity'], reverse=True)
            if score_list_filter[0]['similarity'] > threshold:
                user_advices = score_list_filter[0]['user_advices']
    
        # filter score_list with non empty inherit_advice
        score_list_filter = [s for s in score_list if len(s['inherit_advices']) > 0]
        if len(score_list_filter) > 0:
            score_list_filter = sorted(score_list_filter, key=lambda x: x['similarity'], reverse=True)
            if score_list_filter[0]['similarity'] > threshold:
                inherit_advices = score_list_filter[0]['inherit_advices']

        # sort the score_list by similarity
        score_list = sorted(score_list, key=lambda x: x['similarity'], reverse=True)
        output = score_list[0]
        # if score_list has non empty user_advice, update the user_advice
        if len(user_advices) > 0:
            for user_advice in user_advices:
                if user_advice not in output['user_advices']:
                    output['user_advices'].append(user_advice)
        # if score_list has non empty inherit_advice, update the inherit_advice
        elif len(inherit_advices) > 0:
            for inherit_advice in inherit_advices:
                if inherit_advice not in output['inherit_advices']:
                    output['inherit_advices'].append(inherit_advice)
        if output['similarity'] > threshold:
            return output
        else:
            # remove the gpt response
            output.pop('gpt_response', None)
            return output

    def store_prompt(self, user_prompt, programing_language, user_prompt_hash, prompt_summary, gpt_summary_embedding, gpt_response, inherit_advices = [], user_advices = []):
        # store the infomrmation to folder programing_language,
        prompt_resource = os.path.join(self.prompts_resource, 'prompt_info.yaml')

        if not os.path.exists(self.prompts_resource):
            os.makedirs(self.prompts_resource)
        
        # convert gpt_summary_embedding to string
        embedding_vector = gpt_summary_embedding.numpy().tolist()
        # store the prompt_summary, gpt_summary_embedding in yaml format
        _content_dict = {'prompts':
                            [{'name': user_prompt_hash,
                            'vector': embedding_vector,
                            'content': {
                                        'prompt_summary': prompt_summary,
                                        'user_prompt_hash': user_prompt_hash,
                                        'user_prompt': user_prompt,
                                        'user_advices': user_advices,
                                        'inherit_advices': inherit_advices,
                                        'gpt_response': gpt_response}}]}
        
        # load the prompt_info.yaml
        if os.path.exists(prompt_resource):
            with open(prompt_resource, 'r') as f:
                prompt_info = yaml.load(f, Loader=yaml.RoundTripLoader)
            prompt_info = {} if prompt_info is None else prompt_info
            # handle prompt_info['prompts'] is None
            if prompt_info.get('prompts') is None:
                prompt_info['prompts'] = []
            # handle prompt_info['prompts'] is not None
            else:
                # if the prompt_summary is already in the prompt_info['prompts'], update the prompt_info
                for i, saved_prompt in enumerate(prompt_info['prompts']):
                    if saved_prompt['content']['user_prompt'] == user_prompt:
                        prompt_info['prompts'][i] = _content_dict['prompts'][0]
                # if the prompt_summary is not in the prompt_info['prompts'], update the prompt_info
                else:
                    prompt_info['prompts'].append(_content_dict['prompts'][0])
        else:
            prompt_info = _content_dict
        # store the prompt_info.yaml
        with open(prompt_resource, 'w') as f:
            # dump the prompt_info to yaml file, folded style
            yaml.dump(prompt_info, f, Dumper=yaml.RoundTripDumper)

    def load_python_viz_example(self):
        file_path = "examples/streamlit_viz_example.py"
        file_path = os.path.join(current_dir, file_path)
        # open the python sample file
        with open(file_path, 'r') as f:
            python_sample = f.read()
        return python_sample, file_path
    
    def get_gpt_prompt(n):
        if n == 0.1:
            return """
user prompt:
##user_prompt##

- The SQL dialects is Snowflake SQL, try to use basic syntax (if possible) for compatibility
- Code only, no text/comments
- Use linebreak to make the script more readable
- if there is a date/time field, sort by date/time
##prompt_advice##

- for forecast/predict request, use this formula
PARSE_JSON(PREDICT_SALES(object_construct('STORE_DEPT_PK', store_dept_pk,'ds', week_date )))predics,
predics:yhat::float weekly_sale_predict,
predics:yhat_upper::float weekly_sale_predict_upper,
predics:yhat_lower::float weekly_sale_predict_lower

- Following JSON is database schema, take this as your source
[{"column_name":"STORE_DEPT_PK","data_type":"TEXT","table_name":"SALES","table_schema":"SNOWPARK"},{"column_name":"UNEMPLOYMENT","data_type":"TEXT","table_name":"SALES","table_schema":"SNOWPARK"},{"column_name":"WEEKLY_SALES","data_type":"FLOAT","table_name":"SALES","table_schema":"SNOWPARK"},{"column_name":"STORE","data_type":"NUMBER","table_name":"SALES","table_schema":"SNOWPARK"},{"column_name":"CPI","data_type":"TEXT","table_name":"SALES","table_schema":"SNOWPARK"},{"column_name":"DEPT","data_type":"NUMBER","table_name":"SALES","table_schema":"SNOWPARK"},{"column_name":"WEEK_DATE","data_type":"TIMESTAMP_NTZ","table_name":"SALES","table_schema":"SNOWPARK"},{"column_name":"TEMPERATURE","data_type":"FLOAT","table_name":"SALES","table_schema":"SNOWPARK"},{"column_name":"ISHOLIDAY","data_type":"BOOLEAN","table_name":"SALES","table_schema":"SNOWPARK"},{"column_name":"STORE_DEPT_PK","data_type":"TEXT","table_name":"SALES","table_schema":"SNOWPARK"},{"column_name":"FUEL_PRICE","data_type":"FLOAT","table_name":"SALES","table_schema":"SNOWPARK"}]

######

WITH temp_cte as ("""

        elif n == 1:
            return """
# output in python function - streamlit_viz_##i##(st: st, df: pd.DataFrame), provide plotly visualisation
# code only, no text/comments
##previous_chart_type##
##force_chart_type##
# hint: use boxplot for categorical data, linechart for timeseries, barchart for ranking
# hint 2: for line charts, include multiple measure fields for better comparison
# provide annotations about the chart content, along with the visualisation
# use provided df from function parameter
# following lines are df.head(n).to_dict(), use as your baseline assumption
# ##df_head##

######

import streamlit as st, pandas as pd, plotly.express as px, plotly as py

def streamlit_viz_##i##(df: pd.DataFrame):"""

        elif n == 1.1:
            return """
- complete your answer in python, return plotly visualisation
- code only, no text/comments
- expected to have similar quality like this example, update python script with suitable reference from data frame
##prompt_advice##

- following lines are df.head(n).to_dict(), use as your baseline assumption
##df_head##
- data type of df
##df_dtype##

######

import streamlit as st, pandas as pd, plotly.express as px, plotly as py, plotly.graph_objects as go

def add_trace(fig, x, y, name, mode, line=None, fill=None):
        trace = go.Scatter(x=x, y=y, name=name, mode=mode, line=line, fill=fill)
        fig.add_trace(trace)

def streamlit_viz_##i##(df: pd.DataFrame):"""

        elif n == 2:
            return """
##python_code##

######

what is the chart type of given python function?

Answer: The chart type of the given python function is"""
        # ask GPT3 to correct user sql syntax
        elif n == 3:
            return """
- correct the following Snowflake SQL syntax, if any:
##user_prompt##


- if the syntax is correct, answer with 'correct'
- code only, no text/comments

######

"""
        # ask GPT3 to classify the user requests
        elif n == 4:
            return """
- provide the programming language that can be used to achieve the following user prompt
- note that usually python is only used for data visualisation, not data processing
- answer one language only, if there are multiple languages, answer with the most significant one

user prompt:
##user_prompt##


######
programing_language:"""
        elif n == 5:
            return """
- provide a summary about given user prompt
- no need to provide code, only text/comments
- your summary will be used as an input for another generative model, so make sure it fits for that purpose

user prompt:
##user_prompt##

######

Assist user to"""