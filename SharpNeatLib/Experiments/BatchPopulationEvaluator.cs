using System;
using SharpNeatLib.Evolution;
using SharpNeatLib.NeuralNetwork;

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
	public class BatchPopulationEvaluator : IPopulationEvaluator
	{
		public INetworkEvaluator networkEvaluator;
		public IActivationFunction activationFn;
		public ulong evaluationCount=0;
    public INetwork[] networks;
    public IGenome[] genomes;

		#region Constructor
        public BatchPopulationEvaluator()
        {
        }
		public BatchPopulationEvaluator(INetworkEvaluator networkEvaluator, IActivationFunction activationFn)
		{
			this.networkEvaluator = networkEvaluator;
			this.activationFn = activationFn;
		}

		#endregion

		#region IPopulationEvaluator Members

		public virtual void EvaluatePopulation(Population pop, EvolutionAlgorithm ea)
		{
			// Evaluate in single-file each genome within the population. 
			// Only evaluate new genomes (those with EvaluationCount==0).
			int count = pop.GenomeList.Count;
      genomes = new IGenome[count];
      networks = new INetwork[count];
			for(int i=0; i<count; i++)
			{
				IGenome g = pop.GenomeList[i];
        genomes[i] = g;
        
				// if(g.EvaluationCount!=0)
				// 	continue;

				INetwork network = g.Decode(activationFn);
        g.Fitness = EvolutionAlgorithm.MIN_GENOME_FITNESS;
        networks[i] = network;
				// if(network==null)
				// {	// Future genomes may not decode - handle the possibility.
				// 	g.Fitness = EvolutionAlgorithm.MIN_GENOME_FITNESS;
				// }
				// else
				// {
				// 	g.Fitness = Math.Max(networkEvaluator.EvaluateNetwork(network), EvolutionAlgorithm.MIN_GENOME_FITNESS);
				// }

				// // Reset these genome level statistics.
				g.TotalFitness = g.Fitness;
				g.EvaluationCount = 1;

				// Update master evaluation counter.
				evaluationCount++;
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
				return networkEvaluator.EvaluatorStateMessage;
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
