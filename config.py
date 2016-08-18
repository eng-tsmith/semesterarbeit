'''
Created on May 11, 2016

@author: Mohamed.Ezz

This config file configures the validation pipeline tasks

A pipeline is composed of the following components :

InputIterator 	-- iterates through input file paths. Each input is assumed to be completely independent of other inputs
Preprocessor 	-- preprocess one input
Predictor		-- runs a prediction model on an input
Postprocessor	-- postprocess predictions
Evaluator		-- evaluates prediction vs ground truth (produces metrics)
Reporter		-- aggregates and writes metrics of different inputs into some report format (e.g., CSV or Excel)

How to use the validation framework :
1- Create classes for all 6 tasks. Each of your classes inherit from one of the *Task classes in validation_task.py
    - For an example of these classes, look at saratan/validation/pipeline/example.py
2- Edit this config file so that Pipeline.InputIterator points to your inputIterator task "class", and same for the other tasks
3- Your task classes must implement the run method and optionally the save method
5- Run python saratan/validation/pipeline/validate.py

Notes:
1- validate.py will pass the output of each step, as an input to the next step, without really checking what it is.
    with the exception that Reporter takes also the output of InputIterator
    , it is the user's responsibility to make sure the output of one step matches the input of the next step
    , but (only) as a convention, input types should be as follows for a typical validation pipeline :
        InputIterator : no input
        Preprocessor  : takes an input tuple (input identifier, img path, ground truth path)
        Predictor     : takes list of 2 volumes [image volume, ground truth volume]
        Postprocessor : takes list of 2 volumes [probability volume, ground truth volume]
        Evaluator     : takes a list of 2 volumes [hard label volume, ground truth volume]
        Reporter      : input tuple (from InputIterator) AND a list of metrics, e.g., [Dice, RVD, ASD]
3- You can save the intermediate results by setting [taskname]_save_to_disk=True in this config file.
    but then the task must implement the save method
4- As a convention, each task class should be defined in a separate file located in the respective directory
    under saratan/validate/pipeline/yourPipelineName/{inputiterator.py or preprocessor.py or....etc}
5- The code controlling the flow of the pipeline is in saratan/validation/pipeline/validate.py
'''

# for example use this
# import example

# import validation_task as validation_task

# import IAM pipeline
from IAM_pipeline.inputiterator import IAM_InputIterator
from IAM_pipeline.preprocessor import IAM_Preprocessor
from IAM_pipeline.predictor import IAM_Predictor
from IAM_pipeline.postprocessor import IAM_Postprocessor
from IAM_pipeline.evaluator import IAM_Evaluator
from IAM_pipeline.reporter import IAM_Reporter

class IAM_pipeline:
    InputIterator = IAM_InputIterator
    InputIterator_save_to_disk = False
    InputIterator_save_directory = 'input_iterator_output_directory'

    Preprocessor = IAM_Preprocessor
    Preprocessor_save_to_disk = False
    Preprocessor_save_directory = 'preprocessor_output_directory'

    Predictor = IAM_Predictor
    Predictor_save_to_disk = True
    Predictor_save_directory = 'predictor_output_directory'

    Postprocessor = IAM_Postprocessor
    Postprocessor_save_to_disk = False
    Postprocessor_save_directory = 'postprocessor_output_directory'

    Evaluator = IAM_Evaluator
    Evaluator_save_to_disk = False
    Evaluator_save_directory = 'evluator_output_directory'

    Reporter = IAM_Reporter
    Reporter_save_directory = 'report_output_directory'

# __________________________________
# ________ SELECT PIPELINE _________
# __________________________________

Pipeline = IAM_pipeline



