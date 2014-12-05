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
	public class SingleFilePopulationEvaluator : IPopulationEvaluator
	{

		public INetworkEvaluator networkEvaluator;
		public IActivationFunction activationFn;
		public ulong evaluationCount=0;
    public Func<IGenome, INetwork> genomeDecode;

		#region Constructor
        public SingleFilePopulationEvaluator()
        {
        }
		public SingleFilePopulationEvaluator(INetworkEvaluator networkEvaluator, IActivationFunction activationFn)
		{
			this.networkEvaluator = networkEvaluator;
			this.activationFn = activationFn;
      this.genomeDecode = DefaultGenomeDecoder;//new Func<IGenome, INetwork>();
		}

		#endregion

		#region IPopulationEvaluator Members

    public INetwork DefaultGenomeDecoder(IGenome g) {
      return g.Decode(activationFn);
    }

		public virtual void EvaluatePopulation(Population pop, EvolutionAlgorithm ea)
		{
			// Evaluate in single-file each genome within the population. 
			// Only evaluate new genomes (those with EvaluationCount==0).
			int count = pop.GenomeList.Count;
			for(int i=0; i<count; i++)
			{
				IGenome g = pop.GenomeList[i];
				if(g.EvaluationCount!=0)
					continue;

				INetwork network = genomeDecode(g);
				if(network==null)
				{	// Future genomes may not decode - handle the possibility.
					g.Fitness = EvolutionAlgorithm.MIN_GENOME_FITNESS;
				}
				else
				{
					g.Fitness = Math.Max(networkEvaluator.Evaluate(network), EvolutionAlgorithm.MIN_GENOME_FITNESS);
				}

				// Reset these genome level statistics.
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
