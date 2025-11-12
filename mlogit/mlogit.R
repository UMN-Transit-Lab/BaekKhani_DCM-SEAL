setwd(dirname(rstudioapi::getSourceEditorContext()$path))
library(dplyr)
library(mlogit)

df<-read.csv('dfR.csv')
nalt<-max(df$alt)
dftr<-df %>% filter(flag=='tr') %>% select(-flag)
dfts<-df %>% filter(flag=='ts') %>% select(-flag)

dftr<-mlogit.data(dftr,shape='long',choice='match',alt.var='alt')
dfts<-mlogit.data(dfts,shape='long',choice='match',alt.var='alt')
LL0<-(nrow(dfts)/nalt)*log(1/nalt)

shortMNL<-mlogit(match~x1+x2+x3,dftr)
summary(shortMNL)
smTest<-cbind(dfts[dfts$match==1,c("chid","alt")],data.frame(predict(shortMNL,dfts)))
smTest$lik<-(smTest$alt=='1')*(smTest$X1)+(smTest$alt=='2')*(smTest$X2)
LLS<-sum(log(smTest$lik))
rhoS<-1-LLS/LL0
print(rhoS)

fullMNL<-mlogit(match~x1+x2+x3|s11+s12+s21+s22+e11+e12+e21+e22+o11+o12+o13|0,dftr)
summary(fullMNL)
fmTest<-cbind(dfts[dfts$match==1,c("chid","alt")],data.frame(predict(fullMNL,dfts)))
fmTest$lik<-(fmTest$alt=='1')*(fmTest$X1)+(fmTest$alt=='2')*(fmTest$X2)
LLF<-sum(log(fmTest$lik))
rhoF<-1-LLF/LL0
print(rhoF)

shortreport<-as.data.frame(summary(shortMNL)$CoefTable)
shortreport$MRS<-shortreport$Estimate/shortreport[2,1]
print(shortreport)

fullreport<-as.data.frame(summary(fullMNL)$CoefTable)
fullreport$MRS<-fullreport$Estimate/fullreport[2,1]
print(fullreport)

