using Microsoft.ML.Data;

namespace SentimentWebAPI.Models
{
    public class SentimentData
    {
        [LoadColumn(0)] public string Text { get; set; }
        [LoadColumn(1)]
        public string Label;
    }
   
    public class SentimentPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabel { get; set; }
    }
}
