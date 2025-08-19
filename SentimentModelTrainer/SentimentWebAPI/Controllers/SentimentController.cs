using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;
using SentimentWebAPI.Models;

namespace SentimentWebAPI.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class SentimentController : ControllerBase
    {
        private static readonly Lazy<PredictionEngine<SentimentData, SentimentPrediction>> _predictionEngine
            = new(() =>
            {
                var mlContext = new MLContext();
                var modelPath = Path.Combine(Directory.GetCurrentDirectory(), "SentimentModel.zip");
                var model = mlContext.Model.Load(modelPath, out _);
                return mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);
            });

        [HttpPost]
        public IActionResult Analyze([FromBody] SentimentData input)
        {
            if (string.IsNullOrEmpty(input.Label))
            {
                input.Label = ""; // or "Unknown"
            }

            var prediction = _predictionEngine.Value.Predict(input);
            return Ok(new { input.Text, Sentiment = prediction.PredictedLabel });
        }
    }
}
