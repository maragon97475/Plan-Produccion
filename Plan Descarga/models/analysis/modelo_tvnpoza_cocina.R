# Modelo TVN Cocina
###################
library(caret)
library(dplyr)
library(readxl)
library(psych)
library(Metrics)
library(outliers)

#Working directory
print(getwd())

#Loading data 
df0<-read_xlsx("./models/inputs/TVN_COCINA.xlsx") %>% as.data.frame()
df0[,c(9,12)]<-sapply(df0[,c(9,12)],as.numeric) %>% as.data.frame()
df0[,c(9,12)]<-sapply(df0[,c(9,12)],round,2)

## Agrupando data
analy <- df0  %>%  
  group_by(PLANTA, FECHA_PRODUCCION, HORA_ALIM_COCINA) %>% 
  summarize(TBVN.IC = round(max(TBVN_IC),2),
            TBVN.dw = round(weighted.mean(TBVN_d, Declarado),2),
            Rprom = round(mean(R_POZAS),2), 
            Rpromw = round(weighted.mean(R_POZAS, Declarado),2)) %>% 
  as.data.frame()

## Preprocessing
analy <- analy %>% filter(!is.na(TBVN.IC))
bdout<-analy %>% select("TBVN.IC","TBVN.dw","Rpromw")
bdouts<-cbind(bdout,sapply(bdout,scores))
names(bdouts)<-c("TBVN.IC","TBVN.dw","Rpromw","zTBVN.IC","zTBVN.dw","zRpromw")
bdouts2<-bdouts[abs(bdouts$zTBVN.IC)<2,]
bdouts2$id_rel<-ifelse(bdouts2$TBVN.dw<=bdouts2$TBVN.IC,1,0)
bdouts3<-bdouts2[bdouts2$id_rel==1,]

## Train/test
df_m<-bdouts3[,c(1,2,3)]  
n <- nrow(df_m)
n_train <- round(0.7 * n) 
set.seed(123)
train_indices <- sample(1:n, n_train)
train <- df_m[train_indices, ]  
test <- df_m[-train_indices, ]  
train$diff<-(train$TBVN.IC-train$TBVN.dw)
test$diff<-(test$TBVN.IC-test$TBVN.dw)

## Modelo Lineal - Here the Rpromw is the residence time in hours of poza or the TDC in poza
## Here it will print the coefficient on the Estimate column which is the one used
lm_model <- lm(diff ~ -1 + Rpromw, data =train)
summary(lm_model)

## Getting MAE
pred_diff_train<-predict(lm_model,train)
pred_diff_test<-predict(lm_model,test)
mae_train<-sum(abs(train$diff-pred_diff_train))/nrow(train)
mae_test<-sum(abs(test$diff-pred_diff_test))/nrow(test)
df_metrics<-data.frame(mae_train=mae_train, mae_test=mae_test)
df_metrics
