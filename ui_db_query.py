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

openai_api_key = os.environ.get('OPENAI_API_KEY')



def is_plot(result_json,question):
  plot_prompt = f"""You have this resulting json which is a result of querying a sqlite database:{result_json}
  The question to answer based on this json is: {question}
  
  Can you now suggest if it is possible to make any plot based on this json that would be useful to a user who asked that question? If no meaningful plot is possible, just say no. The graph should not be forced. Only suggest a plot if it would be helpful to the user. If there is only a single value, there is no need of a plot
  Give the answer in the following format:
  {{'is_graph':'yes','graph_type':'bar','x-axis':'name','y-axis':'frequency','x_axis_data':['vivek','ani','hari'],'y_axis_data':[0.2,0.3,0.4]}}
  This final dictionary alone should be the response. Please note that there should be just a dictionary of the above format and nothing else in the response. This dictionary will be directly used for further processing
  is_graph should be yes only if there are 2 parts in the json: one for categories and the other for numeric values.
  For Eg: {{'Productline': {{0: 'Home and lifestyle',
  1: 'Electronic accessories',
  2: 'Sports and travel'}},
 'TotalCount': {{0: 65, 1: 60, 2: 59}}}} =>is_graph:yes
  
  {{'Productline': {{0: 'Home and lifestyle'}}}} =>is_graph:no

  Note: If the resulting json has time series, the chart type should be line chart

  Final for x-axis,y-axis, write descriptions of what is actually on the axis based on the question and dont copy directly from the json
  """
  plot_response = get_chat_response_closed(plot_prompt,"gpt-3.5-turbo-0125")  
  return plot_response

def make_bar_plot(plot_json):
    fig = plt.figure(figsize=(20, 5))
    x_values = plot_json["x_axis_data"]
    y_values = plot_json["y_axis_data"]
    plt.bar(x_values, y_values, color='green', width=0.4)
    plt.xlabel(plot_json["x-axis"])
    plt.ylabel(plot_json["y-axis"])
    return fig  # Return the figure object

def make_line_plot(plot_json):
    fig = plt.figure(figsize=(10, 5))
    x_values = plot_json["x_axis_data"]
    y_values = plot_json["y_axis_data"]
    plt.plot(x_values, y_values)
    plt.xlabel(plot_json["x-axis"])
    plt.ylabel(plot_json["y-axis"])
    return fig  # Return the figure object


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



def get_chat_response_closed(prompt, model, temperature=0.7, max_tokens=150):
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

def get_chat_response(prompt,model):
  openai = OpenAI(
    api_key="F4wCmQnLM8pZVPC9bxgin2wmWNDQZEhn",
    base_url="https://api.deepinfra.com/v1/openai",
)

  chat_completion = openai.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": f"{prompt}"}],
)

  return chat_completion.choices[0].message.content

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
  df = pd.read_csv(csv_file_name)
  for col in df.columns:
    df = df.rename(columns={col:col.replace(" ","")})
  push_df_in_db(df,"my_database.db","my_table")
  table_name = "my_table"
  # question = """which is the most common mode of payment?"""
  stage1_prompt = f"""I am currently working with a single table in a sqlite database. The information about this table are as follows:
Table name: {table_name}
Column Names: {list(df.columns)}
Data type for each column:"""
  for col in list(df.columns):
    stage1_prompt+=f"""\n{col} : {type(df[col][0])}"""
  stage1_prompt+=f"""\nBased on this information, please write sqlite queries for the following question:{question}
Note: THIS IS VERY IMPORTANT. ONLY GIVE A SINGLE SQL QUERY AND NO OTHER INFORMATION IN YOUR ANSWER. THERE SHOULD NOT BE ANYTHING EXCEPT THE QUERY ITSELF.DONT EVEN MENTION THAT IT IS A SQL QUERY, JUST GIVE A SINGLE QUERY. THE OUTPUT WILL BE DIRECTLY EXECUTED ON A SQL SERVER"""

  sql_query = get_chat_response_closed(stage1_prompt,"gpt-3.5-turbo-0125")
  sql_query = sql_query.strip("`")
  sql_query = sql_query.strip()
  sql_query = sql_query.replace("\n"," ")
  stage2_prompt =f"""Here is a SQL Query: {sql_query}. In this sql query, wherever, there is '=' or '<>', extract the value and the corresponding column name and table name and give the response as  [(table,column,value)] pairings. Note that the reponse should contain the pairings as a list of tuples -  one tuple for each pairing and nothing else.

For Eg:
Query: select wkts from batting_table where name in ('tendulkar','sehwag')
Answer: [('batting_table','wkts','tendulkar'),('batting_table','wkts','sehwag')]

Note above that the value is the value equated. It could also be a non equation i.e., <>. Also remember that you only have to give the column name. For eg if there is something like strftime('...',date)='2022-01-03', the column name is just date
Please do not extract any value unless there is '=' or '<>'. This is very important
Query: {sql_query}
Answer:


NOTE: THIS IS VERY IMPORTANT: YOUR ANSWER SHOULD CONTAIN ONLY THE LIST OF TUPLES ITSELF AND NOTHING ELSE. ALL THERE SHOULD BE IN YOUR RESPONSE IS A SINGLE LIST"""

  stage2_response = get_chat_response_closed(stage2_prompt,"gpt-3.5-turbo-0125")  
  stage2_response = stage2_response.strip()
  pairings = get_pairings(stage2_response)
  pairings_new = []
  for i in range(len(pairings)):
    pairings_new.append((pairings[i][0],find_most_common_characters(pairings[i][1],list(df.columns)),pairings[i][2]))

  replacement_values = []
  orig_values = []

  for pair in pairings_new:
    for i in range(len(df[pair[1]])):
      if pair[2].lower() in df[pair[1]][i].lower() and is_date(pair[2].lower())==False:  #no need to replace date values
        replacement_values.append(df[pair[1]][i])
        orig_values.append(pair[2])
  sql_query = sql_query.replace("\n"," ")
  for i in range(len(replacement_values)):
    sql_query = sql_query.replace(orig_values[i],replacement_values[i])

  result_json = get_result_json(sql_query) #Final json result

  stage3_prompt = f"""Here's a question: {question}
This is a json with relevant data that has been extracted by querying a database:{result_json}
Please use the json to answer the question. Please make your answer seem like you are answering the question asked. Dont make any references to the json. That is just for you to deduce the answer to the question
Please understand that you are not chatting with me. Rather, you simply have to answer the question:{question}

Note: IT IS OF UTMOST IMPORTANCE THAT YOU DO NOT MENTION THE JSON AT ALL. ALSO YOU ARE SUPPOSED TO GIVE A VERBAL ANSWER TO THE USER AND NOT WRITE ANY CODE OR GIVE ANY OTHER INSTRUCTION. DIRECTLY ANSWER THE USER'S QUESTION"""
  stage3_response = get_chat_response_closed(stage3_prompt,"gpt-3.5-turbo-0125")
  if result_json:
    plot_json = ast.literal_eval(is_plot(result_json, question))
    if plot_json["is_graph"] == "yes":
        fig = None  # Initialize the figure object
        if plot_json["graph_type"] == "bar":
            fig = make_bar_plot(plot_json)  # Get the figure from the plotting function
        elif plot_json["graph_type"] == "line":
            fig = make_line_plot(plot_json)  # Get the figure from the plotting function
        if fig is not None:
            st.pyplot(fig)  # Display the figure in Streamlit

  return stage3_response

def main():
    st.title('Database Querying Thing Demo')
    
    # File uploader allows user to add their own CSV
    uploaded_file = st.file_uploader("Choose a file", type=['csv'])
    if uploaded_file is not None:
        question = st.text_input("Enter your question:", "")
        if question:
            if st.button('Get Answer'):
                try:
                    # Convert the uploaded file to a dataframe
                    df = pd.read_csv(uploaded_file)
                    # Temporarily save the dataframe to a CSV to use in the existing function
                    temp_csv_name = "temp_uploaded_file.csv"
                    df.to_csv(temp_csv_name, index=False)
                    # Call the existing function to get and display the answer
                    answer = answer_question_on_csv(temp_csv_name, question)
                    st.write(answer)
                except Exception as e:
                    st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
