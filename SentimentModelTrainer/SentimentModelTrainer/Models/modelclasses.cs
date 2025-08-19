using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SentimentModelTrainer.Models
{
    public class SentimentData
    {
        [LoadColumn(0)]
        public string Text;

        [LoadColumn(1)]
        public string Label;
    }

    public class SentimentPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabel;
        public float[] Score;
    }
}
