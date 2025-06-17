
imageDir = 'E:\lance colin y\PROJECT\Alzhiemer detection using PET\Dataset ( new )';
imageSize = [224, 224];  
labelClasses = {'AD', 'CN', 'EMCI', 'LMCI', 'MCI'};  


imds = imageDatastore(fullfile(imageDir, labelClasses), ...
    'LabelSource', 'foldernames', 'IncludeSubfolders', true);

imds.Labels = categorical(imds.Labels);

[imdsTrain, imdsTest] = splitEachLabel(imds, 0.8, 'randomized');

augmenter = imageDataAugmenter('RandRotation', [-10, 10], 'RandXTranslation', [-5, 5], 'RandYTranslation', [-5, 5]);
augmentedImdsTrain = augmentedImageDatastore(imageSize, imdsTrain, 'DataAugmentation', augmenter);
augmentedImdsTest = augmentedImageDatastore(imageSize, imdsTest);

disp('Loading DenseNet-201...');
denseNet = densenet201;

lgraphDenseNet = layerGraph(denseNet);

newLearnableLayerDense = fullyConnectedLayer(numel(labelClasses), ...
    'Name', 'new_fc', 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
lgraphDenseNet = replaceLayer(lgraphDenseNet, 'fc1000', newLearnableLayerDense);

newClassificationLayerDense = classificationLayer('Name', 'new_output');
lgraphDenseNet = replaceLayer(lgraphDenseNet, 'ClassificationLayer_fc1000', newClassificationLayerDense);

numFeatures = 2048; 
numHiddenUnits = 128; 
numClasses = numel(labelClasses);

layersLSTM = [
    sequenceInputLayer(numFeatures, 'Name', 'input')
    bilstmLayer(numHiddenUnits, 'OutputMode', 'last', 'Name', 'bilstm')
    fullyConnectedLayer(numClasses, 'Name', 'fc')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-4, ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augmentedImdsTest, ...
    'ValidationFrequency', 5, ...
    'Verbose', true, ...
    'Plots', 'training-progress');


disp('Training DenseNet-201 model...');
trainedNetDenseNet = trainNetwork(augmentedImdsTrain, lgraphDenseNet, options);


disp('Extracting features from DenseNet-201...');
featuresTrain = activations(trainedNetDenseNet, augmentedImdsTrain, 'fc1000', 'OutputAs', 'rows');
featuresTest = activations(trainedNetDenseNet, augmentedImdsTest, 'fc1000', 'OutputAs', 'rows');


disp('Training Bi-LSTM model...');
trainedLSTM = trainNetwork(featuresTrain, imdsTrain.Labels, layersLSTM, options);


save('DenseNet201_Model.mat', 'trainedNetDenseNet');
save('BiLSTM_Model.mat', 'trainedLSTM');
disp('Trained models saved.');


disp('Testing the models...');
predictedLabelsDenseNet = classify(trainedNetDenseNet, augmentedImdsTest);
predictedLabelsLSTM = classify(trainedLSTM, featuresTest);

testLabels = imdsTest.Labels;
accuracyDenseNet = sum(predictedLabelsDenseNet == testLabels) / numel(testLabels);
accuracyLSTM = sum(predictedLabelsLSTM == testLabels) / numel(testLabels);

disp(['DenseNet-201 Accuracy: ', num2str(accuracyDenseNet)]);
disp(['Bi-LSTM Accuracy: ', num2str(accuracyLSTM)]);


figure;
confusionchart(testLabels, predictedLabelsDenseNet, 'Title', 'Confusion Matrix - DenseNet-201');
figure;
confusionchart(testLabels, predictedLabelsLSTM, 'Title', 'Confusion Matrix - Bi-LSTM');


disp('Calculating ROC curve...');
[~, scoresDenseNet] = classify(trainedNetDenseNet, augmentedImdsTest);
[~, scoresLSTM] = classify(trainedLSTM, featuresTest);
classes = unique(testLabels);

figure;
hold on;
for i = 1:numel(classes)
    [X, Y, ~, AUC] = perfcurve(double(testLabels == classes(i)), scoresDenseNet(:, i), 1);
    plot(X, Y, 'DisplayName', ['DenseNet-201 ', char(classes(i)), ' (AUC = ', num2str(AUC), ')']);
end
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve - DenseNet-201');
legend('Location', 'Best');
hold off;

figure;
hold on;
for i = 1:numel(classes)
    [X, Y, ~, AUC] = perfcurve(double(testLabels == classes(i)), scoresLSTM(:, i), 1);
    plot(X, Y, 'DisplayName', ['Bi-LSTM ', char(classes(i)), ' (AUC = ', num2str(AUC), ')']);
end
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve - Bi-LSTM');
legend('Location', 'Best');
hold off;
