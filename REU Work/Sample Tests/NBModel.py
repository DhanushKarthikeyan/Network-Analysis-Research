
#play_tennis = pd.read_csv("Forum12data.csv")
from sklearn import tree
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
import csv

dbTraining = []
dbTest = []
X_training = []
Y_training = []
classVotes = []

#reading the data in a csv file
with open('optdigits.tra', 'r') as trainingFile:
  reader = csv.reader(trainingFile)
  for i, row in enumerate(reader):
      dbTraining.append (row)

#reading the data in a csv file
with open('optdigits.tes', 'r') as testingFile:
  reader = csv.reader(testingFile)
  for i, row in enumerate(reader):
      dbTest.append (row)
      classVotes.append([0,0,0,0,0,0,0,0,0,0])

  print("Started my base and ensemble classifier ...")

  correct_predictions = 0
  incorrect_predictions = 0

  for k in range(20):

      bootstrapSample = resample(dbTraining, n_samples=len(dbTraining), replace=True)

      for trainingSample in bootstrapSample:
          X_training.append(trainingSample[:-1])
          Y_training.append(trainingSample[-1])

      #fitting the decision tree to the data
      clf = tree.DecisionTreeClassifier(criterion = 'entropy')
      clf = clf.fit(X_training, Y_training)

      for i, testSample in enumerate(dbTest):

          class_predicted = clf.predict([testSample[:-1]])[0]
          classVotes[i][int(class_predicted)] += 1

          if k == 0:
             true_label = int(testSample[-1:][0])
             if int(class_predicted) == true_label:
                correct_predictions += 1
             else:
                incorrect_predictions += 1

      if k == 0:
         accuracy = correct_predictions/(correct_predictions + incorrect_predictions)
         print("Finished my base classifier (fast + relatively low accuracy) ...")
         print("My base classifier accuracy: " + str(accuracy))
         print("")

  correct_predictions = 0
  incorrect_predictions = 0

  for i, testSample in enumerate(dbTest):

      class_predicted_ensemble = classVotes[i].index(max(classVotes[i]))
      true_label = int(testSample[-1:][0])

      if class_predicted_ensemble == true_label:
         correct_predictions += 1
      else:
         incorrect_predictions += 1

  accuracy = correct_predictions/(correct_predictions + incorrect_predictions)

  print("Finished my ensemble classifier (slow + higher accuracy) ...")
  print("My ensemble accuracy: " + str(accuracy))
  print("")

  print("Started Random Forest algorithm ...")

  #Create a Random Forest Classifier
  clf=RandomForestClassifier(n_estimators=20)

  #Fit Random Forest to the training data
  clf.fit(X_training,Y_training)

  correct_predictions = 0
  incorrect_predictions = 0

  for testSample in dbTest:
      class_predicted_rf = int(clf.predict([testSample[:-1]])[0])
      true_label = int(testSample[-1:][0])

      if class_predicted_rf == true_label:
         correct_predictions += 1
      else:
         incorrect_predictions += 1

  accuracy = correct_predictions/(correct_predictions + incorrect_predictions)

  print("Random Forest accuracy: " + str(accuracy))
  print("Finished Random Forest algorithm (much faster and the highest accuracy!) ...")




