# to read dataset from data source, it could be from live stream, databases like mongodb, hadoop, local system, etc

# we would read in the data
# we would split the data into train and test
# we are ingsting from our notebook folder, and creating an "artifact folder" inside the component folder
# to save the raw data, we are also spliting from the raw data into "train and test"
# and creating a files called "raw.csv, test.csv and train.csv" to save the raw data, the splitted train and test data into the "artifacts folder"
# we worked on jupyter and have our data and notebook files saved in the notebook folder
# since we want to make our code modular and be able to create pipeines and modules we are ingesting from the notebook folder into our "src -> component folder" which have all our source code
# we are then creating folder called "artifacts" and the files name, and ingesting the data from the notebook folder into our already created folder in our "src"


import os
import sys  #we need this for our custom exception
from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


#creating our folder and file names inside the src folder, inside the component folder
# with the dataclass decoration, we can directly define our class variables like " train_data_path"
# without the @dataclass , inside a class, to define the class variable, you basically use "__init__" but if you use the @dataclass, you will be able to directly define class variables
# we will create an artifacts folder so we can see all our output, 
# so we are giving our dataingestion component a path , and all our files will be saved in this path, the train.csv data will be saved in this path
# all the output will be saved in the artifacts folder, the output is any file saved in a folder or a numpy arrary, in this case, out output is the traim.csv, test.cst and raw.csv
# this is the input we are giving (train_data_path: str = os.path.join('artifacts',"train.csv"))
# and the data ingestion will save the train.csv in the (train_data_path:) path


# "train_data_path: str = os.path.join('artifacts',"train.csv")" this is the path or "input" we are giving to our data ingestion component
# and the data ingestion component wiil save all the files(train,test,data) in the above path inside the "artifact folder"
# all the output will be saved in the "artifacts folder"

@dataclass
class dataingestionconfig:
    train_data_path: str = os.path.join('artifacts',"train.csv")
    test_data_path: str = os.path.join('artifacts',"test.csv")
    raw_data_path: str = os.path.join('artifacts',"data.csv")




# starting our class; 1st step. creating a variable name called;"ingestion_comfig"  as assigning the ataingestionconfig to it
class dataingestion:
    def __init__(self):
        self.ingestion_config = dataingestionconfig()


 #second step; to read the dataset   
    def initiate_data_ingestion(self):
        logging.info("entered the data ingestion method or component")
        try:
            df = pd.read_csv('notebook\data\stud.csv')  #here you can read your data from apis, mongodb or any other souces, this is there you read the data from
            
            logging.info('has read the dataset as dataframe')
 
 
 # we already know the path of the train data, the path will be "artifacts\train.csv
 # the "artifacts" is a folder, so lets go ahead to create our "artifact" folder with the help of train,test,raw data path
 #  we would be using "os.makedirs...." to do so, and inside the "ingestion_config" we have a parameter called the "train_data_paht"
 #  inside the "makedire", we would combine the directory path name, ;"os.path.dirname" with <- this we are getting the directory name with respect to the specific path
 #  "exist_ok=True" this mean if the path is there, which is the "train_data_path" we would keep that particular folder, we dont have to delect it and again create it
 #to check if there is a dir name and if it is there keep it folder so we dont have to delect and recreate again           
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)



#saving the raw data to csv  to the same path which is the "ingestion_config"    
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)


# splitting the dataset , this is very important to split for ci/cd and pipeline, when new data comes it would automatically split
# we are spliting the dataset and saving the dataset in the "artifacts folder"           
            logging.info("train test split initiated")
            train_set,test_set = train_test_split(df, test_size = 0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("ingestion of the data is completed")

 
# for the return, and the below specified path, we would be needing it for our "data_transformation"
# so, we would pass this code,in the next step of data transformation, so that the data transformation, will just grab this information and take all these data point and start the transformation process          
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        
        except Exception as e:
            raise CustomException(e,sys)
        

# to initiate and run
# ones we excute this code, it will create our "artifacts folder" which can be visible in our directory
# it will also create a log file
# open terminal 

if __name__=="__main__":
    obj  = dataingestion()
    train_data,test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data,test_data)


    ModelTrainer = ModelTrainer()
    print(ModelTrainer.initiate_model_trainer(train_arr, test_arr))



    