using System;
using System.Collections.Generic;
using System.Text;
using SharpNeatLib.NeuralNetwork;
using UnityEngine;

namespace SharpNeatLib.Experiments
{
    public class HyperNEATParameters
    {
        public static double threshold = 0;
        public static double weightRange = 0;
        public static int numThreads = 0;
        public static IActivationFunction substrateActivationFunction = null;
        public static System.Collections.Generic.Dictionary<string, double> activationFunctions = new Dictionary<string, double>();
        public static System.Collections.Generic.Dictionary<string, string> parameters = new Dictionary<string, string>();
        static HyperNEATParameters()
        {
            loadParameterFile();
        }

        public static void loadParameterFile()
        {
          Debug.Log("loadParameterFile()");
          System.Console.WriteLine("loadParameterFile() console");
            try
            {
                System.IO.StreamReader input = new System.IO.StreamReader(@"params.txt");
                string[] line;
                double probability;
                bool readingActivation = false;
                while (!input.EndOfStream)
                {
                    line = input.ReadLine().Split(' ');
                    if (line[0].Equals("StartActivationFunctions"))
                    {
                        readingActivation = true;
                    }
                    else if (line[0].Equals("EndActivationFunctions"))
                    {
                        readingActivation = false;
                    }
                    else
                    {
                        if (readingActivation)
                        {
                            double.TryParse(line[1], out probability);
                            activationFunctions.Add(line[0], probability);
                        }
                        else
                        {
                            parameters.Add(line[0].ToLower(), line[1]);
                        }
                    }
                }
            }
            catch (Exception e)
            {
                System.Console.WriteLine(e.Message);
                System.Console.WriteLine("Error reading params.txt file, check file location and formation");
                //close program
            }
            ActivationFunctionFactory.setProbabilities(activationFunctions);

            setParameterDouble("threshold", ref threshold);
            setParameterDouble("weightrange", ref weightRange);
            setParameterInt("numberofthreads", ref numThreads);
            setSubstrateActivationFunction();
        }

        private static void setSubstrateActivationFunction()
        {
            string parameter=getParameter("substrateactivationfunction");
            if(parameter!=null)
                substrateActivationFunction=ActivationFunctionFactory.GetActivationFunction(parameter);
        }

        public static string getParameter(string parameter)
        {
            if (parameters.ContainsKey(parameter))
                return parameters[parameter];
            else
                return null;
        }

        public static void setParameterDouble(string parameter, ref double target)
        {
            parameter = getParameter(parameter.ToLower());
            if (parameter != null)
                double.TryParse(parameter, out target);
        }

        public static void setParameterInt(string parameter, ref int target)
        {
            parameter = getParameter(parameter.ToLower());
            if (parameter != null)
                int.TryParse(parameter, out target);
        }
    }
}
