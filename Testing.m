function testing1()
    % Create GUI Figure
    fig = figure('Name', 'Alzheimer Detection System', 'NumberTitle', 'off', ...
                 'Position', [200, 100, 900, 600], 'Color', [0.95 0.95 0.95]);

    % Load Models
    mat_data_dense = load('DenseNet201_Model.mat');
    denseNetModel = mat_data_dense.trainedNetDenseNet;
    
    mat_data_lstm = load('BiLSTM_Model.mat');
    lstmModel = mat_data_lstm.trainedLSTM;

    % UI Layout
    uicontrol('Style', 'pushbutton', 'String', 'Load Image', ...
              'Position', [50, 540, 120, 40], 'FontSize', 12, ...
              'Callback', @uploadImage, 'BackgroundColor', [0.7 0.8 1]);
          
    uicontrol('Style', 'pushbutton', 'String', 'Predict', ...
              'Position', [180, 540, 120, 40], 'FontSize', 12, ...
              'Callback', @predictImage, 'BackgroundColor', [0.7 1 0.7]);

    % Image Display
    uicontrol('Style', 'text', 'String', 'Input Image', ...
              'Position', [100, 490, 100, 20], 'FontSize', 10, ...
              'FontWeight', 'bold', 'BackgroundColor', [0.95 0.95 0.95]);
    axesImage = axes('Units', 'pixels', 'Position', [50, 270, 300, 200]);

    % Prediction Result Display
    finalPredictionText = uicontrol('Style', 'text', 'String', '', ...
                              'Position', [50, 230, 800, 40], 'FontSize', 14, ...
                              'FontWeight', 'bold', 'ForegroundColor', [0 0 0], ...
                              'BackgroundColor', [0.95 0.95 0.95], 'HorizontalAlignment', 'left', ...
                              'FontUnits', 'normalized');

    % Prediction Probability Graph
    uicontrol('Style', 'text', 'String', 'Prediction Confidence', ...
              'Position', [530, 490, 200, 20], 'FontSize', 10, ...
              'FontWeight', 'bold', 'BackgroundColor', [0.95 0.95 0.95]);
    axesGraph = axes('Units', 'pixels', 'Position', [400, 270, 450, 200]);

    % Table for Displaying Results
    uicontrol('Style', 'text', 'String', 'Prediction Results', ...
              'Position', [400, 230, 150, 20], 'FontSize', 10, ...
              'FontWeight', 'bold', 'BackgroundColor', [0.95 0.95 0.95]);

    tableResult = uitable('Position', [400, 100, 450, 120], ...
                          'ColumnName', {'Model', 'Predicted Class', 'Confidence (%)'}, ...
                          'RowName', [], 'FontSize', 12);

    % Global Image Storage
    global img;

    function uploadImage(~, ~)
        [file, path] = uigetfile({'*.jpg;*.png;*.jpeg', 'Image Files'});
        if isequal(file, 0)
            return;
        end
        imgPath = fullfile(path, file);
        img = imresize(imread(imgPath), [224, 224]);
        imshow(img, 'Parent', axesImage);
    end

    function predictImage(~, ~)
        if isempty(img)
            errordlg('Please upload an image first!', 'Error');
            return;
        end
        
        % Feature Extraction and Prediction
        featuresTest = activations(denseNetModel, img, 'avg_pool', 'OutputAs', 'rows');
        featuresTestCell = {featuresTest'};
        
        denseNetPrediction = predict(denseNetModel, img);
        lstmPrediction = predict(lstmModel, featuresTestCell);
        combinedPrediction = (denseNetPrediction + lstmPrediction) / 2;

        classLabels = {'AD', 'CN', 'EMCI', 'LMCI', 'MCI'};
        [~, combinedClass] = max(combinedPrediction);
        maxCombined = max(combinedPrediction);

        % Update Table
        tableResult.Data = {'Combined', classLabels{combinedClass}, sprintf('%.2f%%', maxCombined * 100)};

        % Display Final Prediction
        finalPredictionText.String = sprintf('Final Diagnosis: %s (Confidence: %.2f%%)', ...
                                              classLabels{combinedClass}, maxCombined * 100);
        
        % Plot Predictions
        axes(axesGraph);
        bar(categorical(classLabels), combinedPrediction, 'FaceColor', [0.2 0.6 1]);
        title('Combined Model Prediction Probabilities', 'FontSize', 12, 'FontWeight', 'bold', 'Color', [0 0 0]);
        ylabel('Probability');
        ylim([0 1]);
        grid on;
    end
end