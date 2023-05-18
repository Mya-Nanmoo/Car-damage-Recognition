# Data Source

- Car Detection: CFAR-10 download and config automated.
- Damage Detection:
  Instructions to setup
  > - Download the data from this Link: https://www.kaggle.com/datasets/anujms/car-damage-detection?resource=download
  > - Extract the data.
  > - Configure the paths in train_damage_det.py file eith respective training and validation paths.
  > - Thats it.

# Project configuration

- Make sure that you install all the required libraries in requirement.txt file. use this command to install all libraries.
  > `pip install -r requirements.txt`
- Make sure you run train the models first before running the project. If you dont have a dedicated GPU in your computer, use Google Colab, I also create notebook just incase. After running notebook download the models created then paste them in 'resources/models' directory in the application.

# Running the Project

- To run the Project. Navigate to the app directory then run the command.
  > `flask --app app run`

### **Thats It Good luck with your project.** !!

**CAUTION!!**

> - I trained the car detector model with data containing boats,cars, trucks, and aeroplanes. It works well when identifying cars. This limited scope might produce bias time to time.
> - I have not included the model in the submission file. Please train the model either using the two train scripts or Jupyter notebook file I have provided. This is because the combined size of the models and the app exceed the maximum submission size of this platform and any any file transfer outside the platform is strictly forbiden.
