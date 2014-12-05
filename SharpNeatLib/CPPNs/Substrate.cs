using System;
using System.Collections.Generic;
using System.Text;
using SharpNeatLib.NeuralNetwork;
using SharpNeatLib.NeatGenome;
using SharpNeatLib.Experiments;

namespace SharpNeatLib.CPPNs
{
  public class Substrate
  {
    public uint inputCount;
    public uint outputCount;
    public uint hiddenCount;

    public float inputDelta;
    public float hiddenDelta;
    public float outputDelta;

    public double threshold;
    public double weightRange;
    public IActivationFunction activationFunction;
    public NeuronGeneList neurons;
    public int coordinateCount;
        
    public Substrate()
    {
    }
    public Substrate(uint input, uint output, uint hidden, IActivationFunction function)
    {
      weightRange = HyperNEATParameters.weightRange;
      threshold = HyperNEATParameters.threshold;

      inputCount = input;
      outputCount = output;
      hiddenCount = hidden;
      activationFunction = function;

      inputDelta = 2.0f / (inputCount);
      if (hiddenCount != 0)
        hiddenDelta = 2.0f / (hiddenCount);
      else
        hiddenDelta = 0;
      outputDelta = 2.0f / (outputCount);
      coordinateCount = 4;


      //SharpNEAT requires that the neuronlist be input|bias|output|hidden
      neurons=new NeuronGeneList((int)(inputCount + outputCount+ hiddenCount));
      //setup the inputs
      for (uint a = 0; a < inputCount; a++)
      {
        neurons.Add(new NeuronGene(a, NeuronType.Input, activationFunction));
      }

      //setup the outputs
      for (uint a = 0; a < outputCount; a++)
      {
        neurons.Add(new NeuronGene(a + inputCount, NeuronType.Output, activationFunction));
      }
      for (uint a = 0; a < hiddenCount; a++)
      {
        neurons.Add(new NeuronGene(a + inputCount+outputCount, NeuronType.Hidden, activationFunction));
      }
    }

    public INetwork generateNetwork(INetwork CPPN)
    {
      return generateGenome(CPPN).Decode(null); // XXX why no activation function?
    }

    // There are two different indexes being used.  The within-kind
    // index {input, output, and hidden} and the global-index arranged
    // linearly: input|output|hidden.
    
    // This would be a good case for Hungarian notation since they're
    // both uints.
    
    protected bool IsInput(uint node) {
      return node < inputCount;
    }

    protected bool IsOutput(uint node) {
      return node >= inputCount && node < inputCount + outputCount;
    }

    protected bool IsHidden(uint node) {
      return node >= inputCount + outputCount && node < inputCount + outputCount + hiddenCount;
    }

    protected uint indexForHidden(uint hiddenIndex) {
      return hiddenIndex + inputCount + outputCount;
    }

    protected uint indexForOutput(uint outputIndex) {
      return outputIndex + inputCount;
    }

    protected uint indexForInput(uint inputIndex) {
      return inputIndex;
    }

    protected uint IndexForType(uint node) {
      if (IsInput(node))
        return node;
      else if (IsOutput(node))
        return node - inputCount;
      else if (IsHidden(node))
        return node - inputCount - outputCount;
      else
        throw new ArgumentException("Invalid node " + node);
    }

    protected float DeltaForNode(uint node) {
      if (IsInput(node))
        return inputDelta;
      else if (IsOutput(node))
        return outputDelta;
      else if (IsHidden(node))
        return hiddenDelta;
      else
        throw new ArgumentException("Invalid node " + node);
    }

    protected virtual void SetCoordinates(float[] coordinates,
                                          uint fromNode,
                                          uint toNode) {
      uint i = IndexForType(fromNode);
      float delta = DeltaForNode(fromNode);
      coordinates[0] = -1f + delta / 2.0f + i * delta;
      if (IsInput(fromNode)) {
        coordinates[1] = -1f;
      } else if (IsHidden(fromNode)) {
        coordinates[1] = 0f;
      } else if (IsOutput(fromNode)) {
        coordinates[1] = 1f;
      }

      i = IndexForType(toNode);
      delta = DeltaForNode(toNode);
      coordinates[2] = -1f + delta / 2.0f + i * delta;
      if (IsInput(toNode)) {
        coordinates[3] = -1f;
      } else if (IsHidden(toNode)) {
        coordinates[3] = 0f;
      } else if (IsOutput(toNode)) {
        coordinates[3] = 1f;
      }
    }

    protected float Weight(double output) {
      return (float)(((Math.Abs(output) - (threshold)) / (1f - threshold)) * weightRange * Math.Sign(output));
    }

    public virtual NeatGenome.NeatGenome generateGenome(INetwork network)
    {
      float[] coordinates = new float[coordinateCount];
      float output;
      uint connectionCounter = 0;
      int iterations = 2 * (network.TotalNeuronCount
                            - (network.InputNeuronCount
                               + network.OutputNeuronCount)) + 1;
      ConnectionGeneList connections=new ConnectionGeneList();
      if (hiddenCount > 0)
      {
        for (uint input = 0; input < inputCount; input++)
        {
          for (uint hidden = 0; hidden < hiddenCount; hidden++)
          {
            SetCoordinates(coordinates, indexForInput(input), indexForHidden(hidden));
            network.ClearSignals();
            network.SetInputSignals(coordinates);
            network.MultipleSteps(iterations);
            output = network.GetOutputSignal(0);

            if (Math.Abs(output) > threshold) {
              connections.Add(new ConnectionGene(connectionCounter++, indexForInput(input), indexForHidden(hidden), Weight(output)));
            }
          }
        }
        for (uint hidden = 0; hidden < hiddenCount; hidden++)
        {
          for (uint outputs = 0; outputs < outputCount; outputs++)
          {
            SetCoordinates(coordinates, indexForHidden(hidden), indexForOutput(outputs));
            network.ClearSignals();
            network.SetInputSignals(coordinates);
            network.MultipleSteps(iterations);
            output = network.GetOutputSignal(0);

            if (Math.Abs(output) > threshold)
            {
              connections.Add(new ConnectionGene(connectionCounter++, indexForHidden(hidden), indexForOutput(outputs), Weight(output)));
            }
          }
        }
      }
      else
      {
        for (uint input = 0; input < inputCount; input++)
        {
          for (uint outputs = 0; outputs < outputCount; outputs++)
          {
            SetCoordinates(coordinates, indexForInput(input), indexForOutput(outputs));
            network.ClearSignals();
            network.SetInputSignals(coordinates);
            network.MultipleSteps(iterations);
            output = network.GetOutputSignal(0);

            if (Math.Abs(output) > threshold)
            {
              connections.Add(new ConnectionGene(connectionCounter++, indexForInput(input), indexForOutput(outputs), Weight(output)));
            }
          }
        }
      }
      return new SharpNeatLib.NeatGenome.NeatGenome(0, neurons, connections, (int)inputCount, (int)outputCount);
    }
  }
}
