using System;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Microsoft.ML;
using System.Threading.Tasks;

namespace TaxiMachineLearning
{
    class Program
    {
        const string _datapath = @".\Data\taxi-fare-train.csv";
        const string _testdatapath = @".\Data\taxi-fare-test.csv";
        const string _modelpath = @".\Data\Model.zip";

       static void Main(string[] args)
        {
            PredictionModel<TaxiTrip, TaxiTripFarePrediction> model = Train().Result;
            Evaluate(model);

            var prediction = model.Predict(TestTrips.Trip1);
            Console.WriteLine("Predicted fare: {0}, actual fare: 29.5", prediction.fare_amount);
        }

        public static async Task<PredictionModel<TaxiTrip, TaxiTripFarePrediction>> Train()
        {
            var pipeline = new LearningPipeline();
            pipeline.Add(new TextLoader<TaxiTrip>(_datapath, useHeader: true, separator: ","));
            pipeline.Add(new ColumnCopier(("fare_amount", "Label")));
            pipeline.Add(new CategoricalOneHotVectorizer("vendor_id",
                                             "rate_code",
                                             "payment_type"));
            pipeline.Add(new ColumnConcatenator("Features",
                                    "vendor_id",
                                    "rate_code",
                                    "passenger_count",
                                    "trip_distance",
                                    "payment_type"));

            pipeline.Add(new FastTreeRegressor());

            PredictionModel<TaxiTrip, TaxiTripFarePrediction> model = pipeline.Train<TaxiTrip, TaxiTripFarePrediction>();
            await model.WriteAsync(_modelpath);
            return model;
        }
        private static void Evaluate(PredictionModel<TaxiTrip, TaxiTripFarePrediction> model)
        {
            var testData = new TextLoader<TaxiTrip>(_testdatapath, useHeader: true, separator: ",");

            var evaluator = new RegressionEvaluator();
            RegressionMetrics metrics = evaluator.Evaluate(model, testData);
            // Rms should be around 2.795276
            Console.WriteLine("Rms=" + metrics.Rms);

            Console.WriteLine("RSquared = " + metrics.RSquared);

        }

    }
}
