setwd('~/Documents/GeneralAssembly/GADS4/class4/')
batting <- read.csv('Batting.csv',head=T)
players <- read.csv('Master.csv')
salary <- read.csv('Salaries.csv')

batting_salary <- merge(batting, salary)
data <- merge(players, batting_salary)

model_data <- data[, c('HR', 'RBI', 'SB','R', 'G', 'height', 'weight', 'salary', 'yearID')]
model_data <- model_data[complete.cases(model_data),] #This removes any rows that had an NA value

head(model_data)

model <- lm(salary ~ HR + RBI, data= model_data)
summary(model)

training <- model_data[model_data$yearID == 2011,]
test <- model_data[model_data$yearID == 2012,]
model <- lm(log(salary) ~ HR + RBI + R + G, data= training)

mse <- function(x, y) {
  return(mean( (x-y)^2 ))
}

mae <- function(x, y) {
  return(mean( abs(x-y)))
}

test.predict <- predict(model, test)

plot(test.predict,log(test$salary))
mae(test.predict,test$salary)
hist(test$salary)

model_data$Games50 <- (model_data$G > 50)

model.reg <- glmnet(as.matrix(training[,c('HR','RBI')]),as.matrix(training$salary))
test.reg <- predict(model.reg,as.matrix(test[,c('HR','RBI')]),s=model.reg$lambda.min)
head(test.reg)
str(model.reg)