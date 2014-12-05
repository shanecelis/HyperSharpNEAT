using System;
using SharpNeatLib.Evolution;
using SharpNeatLib.NeuralNetwork;
using System.Collections.Generic; //Always a good idea
using System.Linq;
//using UnityEngine;
using Eppy;

namespace SharpNeatLib.Experiments
{
	/// <summary>
	/// An implementation of IPopulationEvaluator that evaluates all new genomes(EvaluationCount==0)
	/// within the population in single-file, using an INetworkEvaluator provided at construction time.
	/// 
	/// This class provides an IPopulationEvaluator for use within the EvolutionAlgorithm by simply
	/// providing an INetworkEvaluator to its constructor. This usage is intended for experiments
	/// where the genomes are evaluated independently of each other (e.g. not simultaneoulsy in 
	/// a simulated world) using a fixed evaluation function that can be described by an INetworkEvaluator.
	/// </summary>
	public class SingleFilePairPopulationEvaluator : IPopulationEvaluator
	{
		public INetworkPairEvaluator networkEvaluator;
		public IActivationFunction activationFn;
		public ulong evaluationCount=0;

		#region Constructor
        public SingleFilePairPopulationEvaluator()
        {
        }
		public SingleFilePairPopulationEvaluator(INetworkPairEvaluator networkEvaluator, IActivationFunction activationFn)
		{
			this.networkEvaluator = networkEvaluator;
			this.activationFn = activationFn;
		}

		#endregion

		#region IPopulationEvaluator Members

    // Metaheuristics pp 116
    // Coevolution, Single-Elimination Tournament Relative Fitness Assessment
		public virtual void EvaluatePopulation(Population pop, EvolutionAlgorithm ea)
		{

			// Evaluate in single-file each genome within the population. 
			// Only evaluate new genomes (those with EvaluationCount==0).
			int count = pop.GenomeList.Count;

      Random rnd = new Random();
      int[] R = Enumerable.Range(0, count).OrderBy(x => rnd.Next()).ToArray();
      int[] Q = new int[count];
      int Rcount = count;
      int Qcount = 0;
      //Dictionary<int, List<double>> fitnesses;
      Dictionary<Tuple<int, int>, FitnessPair> fitnesses;
      fitnesses = new Dictionary<Tuple<int, int>, FitnessPair>();
      int lgCount = (int) Math.Floor(Math.Log((double)count, 2.0));
      for(int i = 0; i < lgCount ; i += 2) {
        System.Array.Copy(R, Q, Rcount);
        Qcount = Rcount;
        Rcount = 0;
        for (int j = 0; j < Qcount - 1; j += 2) {
          IGenome gj = pop.GenomeList[Q[j]];
          IGenome gjp1 = pop.GenomeList[Q[j + 1]];
          INetwork Qj = gj.Decode(activationFn);
          INetwork Qjp1 = gjp1.Decode(activationFn);
          FitnessPair fpair = networkEvaluator.EvaluateNetworkPair(Qj, Qjp1);
          fitnesses[new Tuple<int, int>(Q[j], Q[j+1])] = fpair;
          if (fpair.fitness1 > fpair.fitness2) {
            R[Rcount] = Q[j];
            Rcount++;
          } else {
            R[Rcount] = Q[j+1];
            Rcount++;
          }
        }
      }
      // Assess Fitness
      // mean of the relative fitness.
      double[] fitness = new double[count];
      int[] fitnessCount = new int[count];
      for (int i = 0; i < count; i++) {
        fitness[i] = 0;
        fitnessCount[i] = 0;
      }
      foreach(KeyValuePair<Tuple<int,int>, FitnessPair> item in fitnesses) {
        int j = item.Key.Item1;
        int k = item.Key.Item2;
        //UnityEngine.Debug.Log(j + " vs " + k + " -> (" + item.Value.fitness1 + ", " + item.Value.fitness2 + ")");
        // deal with item j
        double min = EvolutionAlgorithm.MIN_GENOME_FITNESS;
        // fitness[j] += Math.Max(min, (item.Value.fitness1 - item.Value.fitness2));
        // fitness[k] += Math.Max(min, -(item.Value.fitness1 - item.Value.fitness2));
        fitness[j] += item.Value.fitness1;
        fitness[k] += item.Value.fitness2;

        fitnessCount[j]++;
        fitnessCount[k]++;
      }
      //UnityEngine.Debug.Log("fitnessCount " + string.Join(",", fitnessCount.Select(x => x.ToString()).ToArray()));
      for (int i = 0; i < count; i++) {
        fitness[i] /= (double)fitnessCount[i];
        pop.GenomeList[i].Fitness = Math.Max(EvolutionAlgorithm.MIN_GENOME_FITNESS, fitness[i]);
        pop.GenomeList[i].TotalFitness = Math.Max(EvolutionAlgorithm.MIN_GENOME_FITNESS, fitness[i]);
        pop.GenomeList[i].EvaluationCount +=fitnessCount[i];
      }
		}

		public ulong EvaluationCount
		{
			get
			{
				return evaluationCount;
			}
		}

		public string EvaluatorStateMessage
		{
			get
			{	// Pass on the network evaluator's message.
				return "";//networkEvaluator.EvaluatorStateMessage;
			}
		}

		public bool BestIsIntermediateChampion
		{
			get
			{	// Only relevant to incremental evolution experiments.
				return false;
			}
		}

		public bool SearchCompleted
		{
			get
			{	// This flag is not yet supported in the main search algorithm.
				return false;
			}
		}

		#endregion
	}
}
