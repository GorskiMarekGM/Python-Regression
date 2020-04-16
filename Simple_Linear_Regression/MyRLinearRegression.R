#Simple Linear Regression

#Importing the dataset
dataset = read.csv('Salary_Data.csv')

#splitting data
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Fitting Simple Linear Regression to the Training set
regressor = lm(formula = Salary ~ YearsExperience,
               data = training_set)

#Predict the test set results
y_pred = predict(regressor, newdata = test_set)

#visualising the Training set results

#library
# install.packages('ggplot2')
library(ggplot2)

ggplot() + 
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
            colour = 'red') + 
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue')+
  ggtitle('Salary vs Experience(Trainig Set)') + 
  xlab('Years of Experience') + 
  ylab('Salary')
  

#visualising the Test set results

ggplot() + 
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             colour = 'red') + 
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue')+
  ggtitle('Salary vs Experience(Test Set)') + 
  xlab('Years of Experience') + 
  ylab('Salary')



