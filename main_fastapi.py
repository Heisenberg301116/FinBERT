# code "/mnt/c/Users/Sushant Singh/Truminds/Truminds/FinBERT FastAPI/main_fastapi.py"

"""
cd "D:/Projects/Truminds/FinBERT FastAPI" 
uvicorn main_fastapi:app --reload
"""

from fastapi import FastAPI
import uvicorn

app = FastAPI()     # creating fastapi instance

# ===================================================================================================================================================================
'''
cd "/mnt/c/Users/Sushant Singh/Truminds/Truminds/WSL_Python_and_Venv/FinBERT_Python3.9_env/bin"
source activate

python3 "/mnt/c/Users/Sushant Singh/Truminds/Truminds/FinBERT/main.py" 
'''
# ===================================================================================================================================================================
import os
import sys
import subprocess
import logging
import json

logging.getLogger().setLevel(logging.CRITICAL)

# Setting the current working directory path
current_file_directory = os.path.dirname(os.path.abspath(__file__))

os.chdir(current_file_directory)
print("Set the pwd to : ", os.getcwd())

sys.path.append("src")
from main_bm25 import Start_Lucene_Service, Create_Lucene_Indexing, End_lucene_service, Delete_Lucene_Indexing, Free_Port_8085
import data_cleaning_and_processing
from finbert_qa import FinBERT_QA
from chatgpt_answer_generation import Call_OpenAI_Service, Total_tokens



process = ""
lucene_id = ""
finbert_obj = ""
query_id = 0

#=====================================================================================================================================================================

# Starting Lucene Service
@app.post("/Start_Lucene_Service")
def Start_Service(display_logs : bool = False):
    print("Starting Lucene Service...\n")
    global process
    process = Start_Lucene_Service(display_logs)
    #print("Lucene process = =======================> ", process)

#=====================================================================================================================================================================

# Generate training dataset
@app.post("/Dataset_Generation")
def Process_and_Generate_Training_Dataset():
    # Cleaning and processing data
    print("Cleaning and processing data...")
    data_cleaning_and_processing.main()
    
    # Creating Lucene indexing
    Index_docs(True)

    # Generating training dataset
    print("\nGenerating the training dataset...")
    subprocess.run("python3 src/generate_data.py --query_path data/raw/FiQA_train_question_final.tsv --label_path data/raw/FiQA_train_question_doc_final.tsv --doc_index_id {}".format(lucene_id), shell='True')
    print("\nDone All !!!\n")

#=====================================================================================================================================================================

# Creating Lucene indexing of the cleaned and processed docs.tsv file present at data/Lucene_data/docs.tsv
@app.post("/Create_Lucene_Indexing")
def Index_docs(display_message : bool = False):
    print("Creating Lucene indexing for docs.tsv...")
    docs_path = os.getcwd() + "/data/Lucene_data/docs.tsv"
    indexed_docs_path = os.getcwd() + "/data/Lucene_data/Indexed_docs"
    
    print("docs_path = ",docs_path," and indexed docs_path = ", indexed_docs_path)
    id = Create_Lucene_Indexing(docs_path, indexed_docs_path, display_message)
    
    if(id == -1):
        print("Lucene indexing request could not be processed")
        return "NA"
    elif(id == 'NA'):
        print("Failed to create Lucene indexing !!!")
        return "NA"
    else:
        print("Successfully created Lucene index !!! The lucene index id = ", id)
        global lucene_id
        lucene_id = id
        return id
    
#=====================================================================================================================================================================

# Train the model
@app.post("/Train_Model")
def Training_the_Model(epoch : int, batch : int, lucene_id : str):
    subprocess.run("python3 src/train_models.py --model_type 'bert' --train_pickle data/data_pickle/train_set.pickle --valid_pickle data/data_pickle/valid_set.pickle --bert_model_name 'bert-qa' --doc_index_id {} --learning_approach 'pointwise' --max_seq_len 512 --batch_size {} --n_epochs {} --lr 3e-6 --weight_decay 0.01 --num_warmup_steps 10000".format(lucene_id, batch, epoch), shell='True')
    print("Successfully trained the model !!!")

#=====================================================================================================================================================================

# Evaluate the model performance
@app.post("/Evaluate_Model")
def Evaluate_the_Model():
    subprocess.run("python3 src/evaluate_models.py --test_pickle data/data_pickle/test_set.pickle --model_type 'bert' --max_seq_len 512 --bert_finetuned_model 'finbert-qa' --use_trained_model", shell='True')

#=====================================================================================================================================================================

# Loading models/libraries for predictions
@app.post("/Load_Prediction_Resources")
def Load_Prediction_Resources(lucene_id : str, device : str = 'gpu', max_seq_len : int = 512):
    
    config = {'bert_model_name': 'bert-qa',
              'device': device,
              'max_seq_len': max_seq_len,
              'doc_index_id' : lucene_id}

    global finbert_obj
    finbert_obj = FinBERT_QA(config, 'PREDICTION')

#=====================================================================================================================================================================

# Predict relevant chunks from the fine tuned BERT model
@app.post("/Predict_Relevant_Chunks")
def Predict_from_the_Model(query : str, top_k : int = 5):
    global query_id
    query_id += 1
    finbert_obj.search(query, top_k, query_id)

#=====================================================================================================================================================================

# Generate answer using ChatGPT
@app.post("/Generate_Answer")
def Generate_Answer():
    with open("data/predictions/passages.json") as f:
        data = json.load(f)
      
    data_list = data.split('\n\n')
    question = data_list[1]
    candidate_answers = data_list[2:]
    candidate_answers.pop()

    final_answer = ""
    chunks = candidate_answers[0]
    index = 1

    while(True):
        if(index == len(candidate_answers)):
            if(chunks != ""):
                final_answer += '\n' + '\n' + Call_OpenAI_Service(question, chunks)
                break
            else:
                break

        else:
            if(Total_tokens(chunks + '\n' + candidate_answers[index]) < 4095-300):
                chunks += '\n' + '\n' + candidate_answers[index]
                index += 1
            else:
                final_answer += '\n' + '\n' + Call_OpenAI_Service(question, chunks) + "\n\n"
                chunks = candidate_answers[index]
                index += 1
    
    # final_answer.rstrip('\n')   
    # final_answer.rstrip('\n')      
    print("\n******************************************************************************************************************************************************************************************")
    print(final_answer)
    print("\n******************************************************************************************************************************************************************************************")
    return final_answer

#=====================================================================================================================================================================

# Delete Lucene indexing
@app.delete("/Delete_Lucene_Indexing")
def Delete_Indexing(lucene_id : str):
    return Delete_Lucene_Indexing(lucene_id)

#=====================================================================================================================================================================

# Ending Lucene service
@app.post("/End_Lucene_Service")
def Stop_Lucene():
    global process
    if process == "":
        return "Service was never Started !!!"
    else:
        End_lucene_service(process)
        process = ""
        return "Done"

#=====================================================================================================================================================================

# Releasing port no. 8085 from existing lucene service 
@app.post("/Free_Port_8085")
def Free_8085():
    return Free_Port_8085()