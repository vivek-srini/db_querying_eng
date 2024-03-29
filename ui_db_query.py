from sqlalchemy import create_engine,text
import sqlite3
import pandas as pd
import re
import tiktoken
# from openai import OpenAI
import openai
import ast
import streamlit as st
from datetime import datetime
import matplotlib.pyplot as plt
import os
import anthropic
import seaborn as sns
import tempfile
openai.api_key = st.secrets['OPENAI_API_KEY']
client = anthropic.Anthropic(
    api_key=st.secrets['ANTHROPIC_KEY']
)


from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import numpy as np

def preprocess_features(X):
    """Preprocesses the DataFrame by converting categorical and date features."""
    # Convert date columns to multiple numeric columns (year, month, day)
    for col in X.select_dtypes(include=['datetime', 'object']):
        if pd.to_datetime(X[col], errors='coerce').notnull().all():
            X[col] = pd.to_datetime(X[col])
            X[f"{col}_year"] = X[col].dt.year
            X[f"{col}_month"] = X[col].dt.month
            X[f"{col}_day"] = X[col].dt.day
            X.drop(columns=[col], inplace=True)  # Drop the original date column
    
    # Convert categorical variables to numeric using Label Encoding for simplicity
    # Consider One-Hot Encoding for non-ordinal categorical data
    for col in X.select_dtypes(include=['object', 'category','bool']).columns:
        X[col] = LabelEncoder().fit_transform(X[col])
    
    return X

def compute_feature_importance(df, target_column, features):
    # Ensure the target column and features are in the DataFrame
    if target_column not in df.columns or not all(feature in df.columns for feature in features):
        raise ValueError("Target column or selected features are not in the DataFrame")

    # Select only the relevant columns
    df_relevant = df[[target_column] + features]
    
    # Separate the target variable and the features
    X = df_relevant.drop(columns=[target_column])
    y = df_relevant[target_column]
    
    # Preprocess the features
    X_processed = preprocess_features(X.copy())
    
    # Determine if the target is numeric or categorical for model selection
    target_is_numeric = pd.api.types.is_numeric_dtype(df[target_column])
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    
    # Initialize and fit the model
    model = RandomForestRegressor(random_state=42) if target_is_numeric else RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Compute feature importances
    importances = model.feature_importances_
    feature_importances = pd.DataFrame({'Feature': X_processed.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    
    return feature_importances

def display_feature_importances(feature_importances):
    col1, col2 = st.columns([3, 2])
    with col1:
        st.write("Feature Importances:")
        st.dataframe(feature_importances.style.format({'Importance': '{:.4f}'}))

    with col2:
        st.write("Importance Plot:")
        fig, ax = plt.subplots(figsize=(8, max(6, 0.3 * len(feature_importances))))
        sns.barplot(x='Importance', y='Feature', data=feature_importances, ax=ax)
        plt.tight_layout()
        st.pyplot(fig)




def remove_plural_suffix(text):
    # Pattern to find words ending with (s) optionally followed by an underscore and more characters
    pattern = r'\(s\)(?=_?\w*)'
    # Replace found patterns with nothing, effectively removing them
    result = re.sub(pattern, '', text)
    return result.replace("/","_")

def answer_with_haiku(prompt,sonnet=False):
  model = "claude-3-haiku-20240307"
  if sonnet==True:
      model = "claude-3-sonnet-20240229"
  message = client.messages.create(
    model=model,
    max_tokens=1024,
    messages=[
        {"role": "user", "content": prompt}
    ]
)
  return message.content[0].text

import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def is_categorical(series):
    # Treat a column as categorical if it's a boolean, object, or categorical type
    return series.dtype == bool or pd.api.types.is_categorical_dtype(series) or pd.api.types.is_object_dtype(series)

def analyze_relationship(df, column1, column2):
    if column1 not in df.columns or column2 not in df.columns:
        st.error("Error: Column not found in DataFrame.")
        return

    # Check if each column is categorical according to the defined logic
    col1_is_categorical = is_categorical(df[column1])
    col2_is_categorical = is_categorical(df[column2])
    
    # Scenario 1: One categorical and one numeric column
    if col1_is_categorical != col2_is_categorical:
        numeric_column, categorical_column = (column1, column2) if not col1_is_categorical else (column2, column1)
        
        # Descriptive Statistics display
        st.write(f"Descriptive Statistics of '{numeric_column}' within each '{categorical_column}':")
        descriptive_stats = df.groupby(categorical_column)[numeric_column].describe()
        st.dataframe(descriptive_stats)

        # Visualization setup for dynamic sizing based on the number of categories
        num_categories = len(df[categorical_column].unique())
        plot_width = max(10, num_categories * 1.5)
        plot_height_per_plot = 4
        total_plot_height = plot_height_per_plot * 3 + (2 * num_categories)
        
        fig, axs = plt.subplots(3, 1, figsize=(plot_width, total_plot_height))
        
        # Box Plot
        sns.boxplot(x=categorical_column, y=numeric_column, data=df, ax=axs[0])
        axs[0].set_title('Box Plot', fontsize=20)
        axs[0].tick_params(axis='x', labelsize=14)
        axs[0].tick_params(axis='y', labelsize=14)
        
        # Bar Plot - Showing Mean Values
        mean_values = df.groupby(categorical_column)[numeric_column].mean().reset_index()
        sns.barplot(x=categorical_column, y=numeric_column, data=mean_values, ax=axs[1])
        axs[1].set_title('Bar Plot of Mean Values', fontsize=20)
        axs[1].tick_params(axis='x', labelsize=14)
        axs[1].tick_params(axis='y', labelsize=14)
        
        # Violin Plot
        sns.violinplot(x=categorical_column, y=numeric_column, data=df, ax=axs[2])
        axs[2].set_title('Violin Plot', fontsize=20)
        axs[2].tick_params(axis='x', labelsize=14)
        axs[2].tick_params(axis='y', labelsize=14)
        
        plt.tight_layout()
        st.pyplot(fig)

    # Scenario 2: Both columns are numeric
    elif not col1_is_categorical and not col2_is_categorical:
        correlation = df[[column1, column2]].corr().iloc[0, 1]
        st.write(f"Correlation coefficient between {column1} and {column2}: {correlation:.2f}")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=column1, y=column2, data=df, ax=ax)
        ax.set_title('Scatter Plot', fontsize=20)
        ax.set_xlabel(column1, fontsize=14)
        ax.set_ylabel(column2, fontsize=14)
        plt.tight_layout()
        st.pyplot(fig)

    else:
        st.error("Unsupported combination of column types for analysis.")



# def is_plot(result_json,question):
#   plot_prompt = f"""You have this resulting json which is a result of querying a sqlite database:{result_json}
#   The question to answer based on this json is: {question}
  
#   Can you now suggest if it is possible to make any plot based on this json that would be useful to a user who asked that question? If no meaningful plot is possible, just say no. The graph should not be forced. Only suggest a plot if it would be helpful to the user. If there is only a single value, there is no need of a plot
#   Give the answer in the following format:
#   {{'is_graph':'yes','graph_type':'bar','x-axis':'name','y-axis':'frequency','x_axis_data':['vivek','ani','hari'],'y_axis_data':[0.2,0.3,0.4]}}
#   This final dictionary alone should be the response. Please note that there should be just a dictionary of the above format and nothing else in the response. This dictionary will be directly used for further processing
#   is_graph should be yes only if there are 2 parts in the json: one for categories and the other for numeric values.
#   For Eg: {{'Productline': {{0: 'Home and lifestyle',
#   1: 'Electronic accessories',
#   2: 'Sports and travel'}},
#  'TotalCount': {{0: 65, 1: 60, 2: 59}}}} =>is_graph:yes
  
#   {{'Productline': {{0: 'Home and lifestyle'}}}} =>is_graph:no

#   Note: If the resulting json has time series, the chart type should be line chart

#   Final for x-axis,y-axis, write descriptions of what is actually on the axis based on the question and dont copy directly from the json
#   """
#   plot_response = get_chat_response_closed(plot_prompt,model="gpt-3.5-turbo-0125")  
#   return plot_response

def is_plot(result_json, question):
    from dateutil.parser import parse
    import json

    def is_date(string):
        try:
            parse(string, fuzzy=False)
            return True
        except ValueError:
            return False

    # Convert JSON string to Python dictionary if it's a string
    if isinstance(result_json, str):
        result_json = json.loads(result_json)
    
    # If there's only one column or none, no graph can be made
    if len(result_json.keys()) < 2:
        return {'is_graph': 'no'}

    category_key = None
    numeric_key = None
    time_series_detected = False
    sorted_data = {'x_axis_data': [], 'y_axis_data': []}

    for key in result_json.keys():
        first_value = next(iter(result_json[key].values()))
        if isinstance(first_value, (int, float)):
            numeric_key = key
        else:
            # Attempt to detect if the category is a date/time
            if isinstance(first_value, str) and is_date(first_value):
                time_series_detected = True
                category_key = key
            elif category_key is None:  # Fallback if no date/time detected
                category_key = key

    if category_key and numeric_key:
        graph_type = "line" if time_series_detected else "bar"
        
        # Extract and potentially sort data for plotting
        categories = result_json[category_key]
        values = result_json[numeric_key]
        
        if time_series_detected:
            # Sort by date if it's a time series
            sorted_categories = sorted(categories.items(), key=lambda x: parse(x[1]))
            sorted_values = {k: values[k] for k, _ in sorted_categories}
            sorted_data['x_axis_data'] = [v for _, v in sorted_categories]
            sorted_data['y_axis_data'] = list(sorted_values.values())
        else:
            sorted_data['x_axis_data'] = list(categories.values())
            sorted_data['y_axis_data'] = list(values.values())

        return {
            'is_graph': 'yes',
            'graph_type': graph_type,
            'x-axis': 'Time' if time_series_detected else category_key,
            'y-axis': numeric_key,
            'x_axis_data': sorted_data['x_axis_data'],
            'y_axis_data': sorted_data['y_axis_data']
        }
    
    return {'is_graph': 'no'}




def make_bar_plot(plot_json):
    num_categories = len(plot_json["x_axis_data"])
    # Adjust the width dynamically based on the number of categories
    plot_width = max(10, num_categories * 1.5)  # Base width + extra space per category
    plot_height = 6  # Fixed height, adjust as needed
    
    fig = plt.figure(figsize=(plot_width, plot_height))
    x_values = plot_json["x_axis_data"]
    y_values = plot_json["y_axis_data"]
    plt.bar(x_values, y_values, color='green', width=0.4)
    plt.xlabel(plot_json["x-axis"], fontsize=20)
    plt.ylabel(plot_json["y-axis"], fontsize=20)
    plt.xticks(rotation=45)  # Rotate x-axis labels if necessary
    plt.tick_params(axis='both', which='major', labelsize=14)  # Adjust as needed
    plt.tight_layout()
    return fig

def make_line_plot(plot_json):
    num_points = len(plot_json["x_axis_data"])
    # Dynamically adjust the width based on the number of data points
    plot_width = max(10, num_points * 0.5)  # Base width + extra space per point
    plot_height = 6  # Fixed height, adjust as needed
    
    fig = plt.figure(figsize=(plot_width, plot_height))
    x_values = plot_json["x_axis_data"]
    y_values = plot_json["y_axis_data"]
    plt.plot(x_values, y_values, marker='o')  # Add markers for each data point
    plt.xlabel(plot_json["x-axis"], fontsize=20)
    plt.ylabel(plot_json["y-axis"], fontsize=20)
    plt.xticks(rotation=45)  # Rotate x-axis labels if there are many points
    plt.tick_params(axis='both', which='major', labelsize=14)  # Adjust as needed
    plt.tight_layout()
    return fig


def is_date(string):
    date_formats = [
        "%Y-%m-%d",  # ISO 8601
        "%d-%m-%Y",
        "%m/%d/%Y",
        "%Y/%m/%d",
        "%d/%m/%Y",
        "%B %d, %Y",  # 'January 31, 2020'
        "%b %d, %Y",  # 'Jan 31, 2020'
        "%d %B, %Y",  # '31 January, 2020'
        "%d %b, %Y", 
        "%Y-%m",
        "%Y"
    ]
    
    for date_format in date_formats:
        try:
            datetime.strptime(string, date_format)
            return True  # The string matches this date format
        except ValueError:
            continue  # Try the next format
    
    return False  # None of the formats matched



def get_chat_response_closed(prompt, model, temperature=0.7, max_tokens=500):
    """
    Generate a chat response using OpenAI's chat model.

    Parameters:
    - prompt: The input prompt or message to generate a response for.
    - model: The model to use for generating the response, set to GPT-4 by default.
    - temperature: Controls randomness. Lower is more deterministic. Defaults to 0.7.
    - max_tokens: The maximum number of tokens to generate. Defaults to 150.

    Returns:
    - The generated response as a string.
    """
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "system", "content": "You are a helpful assistant who converts text to sql queries"}, {"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = response.choices[0].message['content']
    return text

def num_tokens_from_string(string: str, encoding_name="cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def push_df_in_db(df,database_path,table_name):
  conn = sqlite3.connect(database_path)
  df.to_sql(table_name, conn, if_exists='replace', index=False)
  conn.close()

def get_result_json(sql_query):
  engine = create_engine(f'sqlite:///my_database.db')
  with engine.begin() as conn:
    query = text(f"""{sql_query}""")
    df = pd.read_sql_query(query, conn)
    return df.to_dict()



def get_pairings(stage2_response):

  input_str = stage2_response.split("\n")[-1]



  try:
    pairings = ast.literal_eval(input_str)
  except ValueError as e:
    pairings = None
    print(f"Error converting string to list of tuples: {e}")
  return pairings

def find_most_common_characters(word, word_list):
    # Initialize maximum count of common characters and the word with the most common characters
    max_common_chars = 0
    most_common_word = None

    # Convert the given word to a set of unique characters for comparison
    word_chars = set(word)

    for word_in_list in word_list:
        # Convert the current word in the list to a set of unique characters
        word_list_chars = set(word_in_list)

        # Find the intersection of characters between the given word and the current word in the list
        common_chars = word_chars.intersection(word_list_chars)

        # Update the maximum count of common characters and the word if this word has more common characters
        if len(common_chars) > max_common_chars:
            max_common_chars = len(common_chars)
            most_common_word = word_in_list

    return most_common_word

#You have a csv. Write csv as a db first to perform queries

def answer_question_on_csv(csv_file_name,question):
  try:
    df = pd.read_csv(csv_file_name)
  except UnicodeDecodeError:
    df = pd.read_csv(csv_file_name,encoding='latin-1')
  for col in df.columns:
    
    df = df.rename(columns={col:remove_plural_suffix(col.replace(" ",""))})
    
  push_df_in_db(df,"my_database.db","my_table")
  table_name = "my_table"
  # question = """which is the most common mode of payment?"""
  stage1_prompt = f"""I am currently working with a single table in a sqlite database. The information about this table are as follows:
Table name: {table_name}\n"""

  stage1_prompt+=f"""Here is information about the columns. I provide the column name, data type of the column and the first 5 entries in each column for you to get an idea of the type of information in the column:"""

  for col in list(df.columns):
      stage1_prompt+=f"""\nColumn Name: {col}\nData type for this column: {df[col].dtype}\nFirst 5 entries in this column:\n {df[col].iloc[:5]}\n\n"""

  stage1_prompt+=f"""\nBased on this information, please write a sqlite query for the following question:{question}
Note: THIS IS VERY IMPORTANT. ONLY GIVE A SINGLE SQL QUERY AND NO OTHER INFORMATION IN YOUR ANSWER. THERE SHOULD NOT BE ANYTHING EXCEPT THE QUERY ITSELF.DONT EVEN MENTION THAT IT IS A SQL QUERY, JUST GIVE A SINGLE QUERY. THE OUTPUT WILL BE DIRECTLY EXECUTED ON A SQL SERVER. ALSO, YOU HAVE TO GIVE EXACTLY 1 SQL QUERY"""
  print(stage1_prompt)
  #sql_query = answer_with_haiku(stage1_prompt)
  sql_query = get_chat_response_closed(stage1_prompt,"gpt-3.5-turbo-0125")
  sql_query = sql_query.strip("`")
  sql_query = sql_query.replace("\n"," ")
  sql_query = sql_query.replace("sql","")
  sql_query = sql_query.strip()
  

  stage2_prompt =f"""Here is a SQL Query: {sql_query}. In this sql query, wherever, there is '=' or '<>', extract the value and the corresponding column name and table name and give the response as  [(table,column,value)] pairings. Note that the reponse should contain the pairings as a list of tuples -  one tuple for each pairing and nothing else.

For Eg:
Query: select wkts from batting_table where name in ('tendulkar','sehwag')
Answer: [('batting_table','wkts','tendulkar'),('batting_table','wkts','sehwag')]

Note above that the value is the value equated. It could also be a non equation i.e., <>. Also remember that you only have to give the column name. For eg if there is something like strftime('...',date)='2022-01-03', the column name is just date
Please do not extract any value unless there is '=' or '<>'. This is very important.If there are no values, just give an empty list
Query: {sql_query}
Answer:


NOTE: THIS IS VERY IMPORTANT: YOUR ANSWER SHOULD CONTAIN ONLY THE LIST OF TUPLES ITSELF AND NOTHING ELSE. ALL THERE SHOULD BE IN YOUR RESPONSE IS A SINGLE LIST. IF THERE ARE NO VALUES, DONT GIVE AN EMPTY STRING - INSTEAD DONT GIVE THE PAIRING AT ALL. ALSO THE VALUES IN THE PAIRINGS SHOULD ALWAYS BE IN QUOTES"""


  stage2_response = get_chat_response_closed(stage2_prompt,"gpt-3.5-turbo-0125")  
  #stage2_response = answer_with_haiku(stage2_prompt)
  #st.write(stage2_response)
  stage2_response = stage2_response.strip()
  pairings = get_pairings(stage2_response)
  pairings_new = []
  for i in range(len(pairings)):
    pairings_new.append((pairings[i][0],find_most_common_characters(pairings[i][1],list(df.columns)),pairings[i][2]))

  replacement_values = []
  orig_values = []

  for pair in pairings_new:
      if not(str(df[pair[1]].dtype)[:3]=="int" or str(df[pair[1]].dtype)[:3]=="float" or str(df[pair[1]].dtype)[:4]=="bool"):
        for i in range(len(df[pair[1]])):
          if str(pair[2]).lower() in str(df[pair[1]][i]).lower() and is_date(str(pair[2]).lower())==False:  #no need to replace date values
            replacement_values.append(df[pair[1]][i])
            orig_values.append(pair[2])
            break
  sql_query = sql_query.replace("\n"," ")
  for i in range(len(replacement_values)):
    sql_query = sql_query.replace(orig_values[i],replacement_values[i])

  result_json = get_result_json(sql_query) #Final json result

  stage3_prompt = f"""Here's a question: {question}
This is a json with relevant data that has been extracted by querying a database:{result_json}
Please use the json to answer the question. Please make your answer seem like you are answering the question asked. Dont make any references to the json. That is just for you to deduce the answer to the question
Please understand that you are not chatting with me. Rather, you simply have to answer the question:{question}

Note: IT IS OF UTMOST IMPORTANCE THAT YOU DO NOT MENTION THE JSON AT ALL. ALSO YOU ARE SUPPOSED TO GIVE A VERBAL ANSWER TO THE USER AND NOT WRITE ANY CODE OR GIVE ANY OTHER INSTRUCTION. DIRECTLY ANSWER THE USER'S QUESTION"""
  stage3_response = get_chat_response_closed(stage3_prompt,"gpt-4")
  #stage3_response = answer_with_haiku(stage3_prompt)
  total_cost = (num_tokens_from_string(stage2_prompt)*0.5 + num_tokens_from_string(stage1_prompt)*0.5+num_tokens_from_string(stage2_response)*1.5+num_tokens_from_string(sql_query)*1.5+num_tokens_from_string(stage3_response)*60)/1000000
   
  
  

  return stage3_response,result_json

def main():
    st.title('Database Querying Thing Demo')

    functionality = st.radio("Choose a functionality:", ('Ask a Question', 'Analyze Relationship','Find Variable Drivers'))

    uploaded_file = st.file_uploader("Choose a file for analysis:", type=['csv'])

    if uploaded_file:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            file_path = tmp_file.name

        df = pd.read_csv(file_path)

        if functionality == 'Ask a Question':
            question = st.text_input("Enter your question:")
            if question and st.button('Submit Question'):
                answer, result_json = answer_question_on_csv(file_path, question)  # Now passing the file path
                #text_area_height = calculate_text_area_height(answer)
                st.write(answer.replace("$","\$"))
                # Additional logic to use result_json as needed
                if result_json:
                    plot_json = is_plot(result_json,question)
                    if plot_json["is_graph"]=="yes":
                        if plot_json["graph_type"]=="bar":
                            fig = make_bar_plot(plot_json)
                            st.pyplot(fig)
                        elif plot_json["graph_type"]=="line":
                            fig = make_line_plot(plot_json)
                            st.pyplot(fig)
                        

        elif functionality == 'Analyze Relationship':
            if uploaded_file is not None:
        # Combine all possible columns for selection, ensuring at least one is numeric for the analysis
                all_columns = df.columns
                column1 = st.selectbox('Select First Column', all_columns, key='first_column_select')
                column2 = st.selectbox('Select Second Column', all_columns, key='second_column_select', index=1 if len(all_columns) > 1 else 0)
        
        # Check if at least one of the selected columns is numeric
                col1_is_numeric = pd.api.types.is_numeric_dtype(df[column1])
                col2_is_numeric = pd.api.types.is_numeric_dtype(df[column2])

                if column1 and column2 and st.button('Analyze Columns', key='analyze_columns_button'):
                    if col1_is_numeric or col2_is_numeric:
                        analyze_relationship(df, column1, column2)
                    else:
                        st.error('At least one of the selected columns must be numeric.')
        elif functionality == 'Find Variable Drivers':
            target_column = st.selectbox('Select Target Column', df.columns, index=0)
            features = st.multiselect('Select Candidate Driver Variables', df.columns.drop(target_column))
            
            if st.button('Compute Feature Importances') and features:
                feature_importances = compute_feature_importance(df, target_column, features)
                display_feature_importances(feature_importances)


if __name__ == "__main__":
    main()

