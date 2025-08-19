using Microsoft.ML;
using SentimentModelTrainer.Models;

var mlContext = new MLContext();

// Load data
var data = mlContext.Data.LoadFromTextFile<SentimentData>(
    "sentiment-data.csv", hasHeader: true, separatorChar: ',');

// Split data
var trainTestSplit = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

// Build pipeline
var pipeline = mlContext.Transforms.Conversion
                    .MapValueToKey("Label")
                .Append(mlContext.Transforms.Text
                    .FeaturizeText("Features", "Text"))
                .Append(mlContext.MulticlassClassification
                    .Trainers.SdcaMaximumEntropy())
                .Append(mlContext.Transforms.Conversion
                    .MapKeyToValue("PredictedLabel"));

// Train the model
var model = pipeline.Fit(trainTestSplit.TrainSet);

// Evaluate
var predictions = model.Transform(trainTestSplit.TestSet);
var metrics = mlContext.MulticlassClassification.Evaluate(predictions);
Console.WriteLine($"Accuracy: {metrics.MacroAccuracy:P2}");

// Save model
mlContext.Model.Save(model, data.Schema, "SentimentModel.zip");
Console.WriteLine("Model saved as SentimentModel.zip");
