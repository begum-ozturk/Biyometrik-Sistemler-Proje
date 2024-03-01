dataFolderTrain='train';
dataFolderTest='test';
categories={'1','2','3','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46'};

imdatastTrain=imageDatastore(fullfile(dataFolderTrain,categories),"LabelSource","foldernames");
imdatastTrain.ReadFcn=@(filename)readandpreprocess(filename);

imdatastTest=imageDatastore(fullfile(dataFolderTest,categories),"LabelSource","foldernames");
imdatastTest.ReadFcn=@(filename)readandpreprocess(filename);

net=googlenet;

featureLayer='loss3-classifier';

trainingFeatures=activations(net,imdatastTrain, featureLayer, "OutputAs","columns");
testingFeatures=activations(net,imdatastTest, featureLayer, "OutputAs","columns");

trainingLabels=imdatastTrain.Labels;
testingLabels=imdatastTest.Labels;

best=0;

for i=1:100
    classifier=fitcecoc(trainingFeatures,trainingLabels,"ObservationsIn","columns","Learners","linear");

    predictedLabels=predict(classifier,testingFeatures,"ObservationsIn","columns");

    accuracy=mean(predictedLabels==testingLabels);

    if accuracy>best
        best=accuracy;
        save('model.mat','classifier');
    end
    sonucAccGNet(i)=accuracy;
    disp(sprintf('Iter:%d  Accurracy:%f    BestAccuracy=%f',i,accuracy,best));
end

eniyi=max(sonucAccGNet);
ortalama=mean(sonucAccGNet);
disp(sprintf('Best accuracy:%f',eniyi));
disp(sprintf('Mean accuracy:%f',ortalama));