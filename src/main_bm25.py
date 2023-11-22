import subprocess
import time
import os
import json
import requests
import signal
import psutil

docs_path, indexed_path = "", ""


def Create_Lucene_Indexing(docs_path, indexed_docs_path, display_message):
    url_create = "http://localhost:8085/createLuceneIndex"
    payload = {
        "inputPath": docs_path,
        "indexPath": indexed_docs_path
    }
    response = requests.put(url_create, json=payload)
    
    if(display_message == True):
        json_obj = json.loads(response.text)
        print("Response status code = ", response.status_code, " and message = ", json_obj['message'])

    if response.status_code == 200:
        json_obj = json.loads(response.text)
        return json_obj['indexId']
        
    else:
        return -1
    


def Search_Top_k_Docs(index_id, query_id, user_query, top_k):
    url_search = "http://localhost:8085/bm25Search"
    payload = {
        "indexId": index_id,
        "queryId": query_id,
        "queryText": user_query,
        "topK": top_k
    }
    response = requests.post(url_search, json=payload)

    if response.status_code == 200:
        json_obj = json.loads(response.text)
        return json_obj['rankedPassageIds']
    else:
        print("Lucene Document Search request failed with status code:", response.status_code)
        return -1



def Delete_Lucene_Indexing(index_id):
    url_delete = " http://localhost:8085/deleteLuceneIndex"

    response = requests.delete(url_delete, data=index_id)
    if response.status_code == 200:
        return "Successfully deleted the given Lucene indexing"
    else:
        print(response.status_code)
        return "Deletion request unsuccessful"


def Start_Lucene_Service(display_logs):      # path is where your FinBERT project is present
    pwd = os.getcwd()
    #print("cwd inside Start_Lucene_service = ", pwd)
    
    # Assigning directory name where .jar file is present
    jar_dir = os.path.join(pwd, 'bm25-lucene-service\\build\\libs')         # for linux, replace '\\' to '/'
    # jar_dir = "'{}'".format(jar_dir)            # to enclose jar_dir with inverted commas (no need to do this for windows else cmd will give error)
    #print("jar dir = ", jar_dir)

    # Framing command for cmd to change directory to jar_dir and execute .jar file
    shell_cmd = 'cd {} && java -jar bm25-lucene-service-1.0-SNAPSHOT.jar'.format(jar_dir)
    
    # Create a subprocess and run the framed command 'shell_cmd'
    if display_logs == False:
        process = subprocess.Popen(shell_cmd, stdout=subprocess.DEVNULL, shell=True)
        time.sleep(15)
        return process
    else:
        process = subprocess.Popen(shell_cmd, shell=True)
        time.sleep(15)
        return process


def End_lucene_service(process):
    print("\n\nGracefully Terminating Lucene Service...")
    # process.send_signal(signal.SIGTSTP)         # for UBUNTU (Ctrl+Z)
    process.send_signal(signal.CTRL_C_EVENT)      # for Windows (Ctrl+C)


def Free_Port_8085():
    pid = None
    for conn in psutil.net_connections():
        if conn.laddr.port == 8085 and conn.status == 'LISTEN':
            pid = conn.pid
        
    if pid == None:
        return 'Port is already free !!!'
    else:
        try:
            subprocess.run(['taskkill', '/F', '/PID', str(pid)], check=True)
            return (f"Process with PID {pid} killed successfully.")
        except subprocess.CalledProcessError as e:
            return(f"Failed to kill process with PID {pid}. Error: {e}")