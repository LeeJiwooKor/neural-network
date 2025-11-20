// nn.js
// Full rewrite: safe numerical ops, pre-activation storage for correct derivatives,
// gradient clipping, memory deduplication, and robust serialization.

class NN {
  constructor({ layers, learningRate = 0.01, gradientClip = 5.0 } = {}) {
    if (!Array.isArray(layers) || layers.length < 2) {
      throw new Error('layers must be an array with at least input and output layer');
    }
    this.layers = layers.map(l => ({ ...l })); // shallow copy
    this.learningRate = Number(learningRate) || 0.01;
    this.gradientClip = Math.max(0, Number(gradientClip) || 0);

    this.weights = [];   // weights[l] is matrix: [outNodes][inNodes]
    this.biases = [];    // biases[l] is column vector: [outNodes][1]
    this.activations = []; // activations[l] for layer l (activation function of layer l > 0)
    this.memory = [];    // array of {input:[], output:[]}

    this._initParameters();
  }

  // ---------------------------
  // Initialization helpers
  // ---------------------------
  _initParameters() {
    for (let i = 0; i < this.layers.length - 1; i++) {
      const inNodes = this.layers[i].nodes;
      const outNodes = this.layers[i + 1].nodes;
      const act = this.layers[i + 1].activation || 'sigmoid';
      this.activations.push(act);
      this.weights.push(this._randomMatrix(outNodes, inNodes, act));
      this.biases.push(this._zeros(outNodes, 1));
    }
  }

  _randomMatrix(rows, cols, activation) {
    let scale = 0.1;
    if (activation === 'relu') scale = Math.sqrt(2 / Math.max(1, cols));
    else if (activation === 'sigmoid' || activation === 'tanh') scale = Math.sqrt(1 / Math.max(1, cols));
    return Array.from({ length: rows }, () =>
      Array.from({ length: cols }, () => (Math.random() * 2 - 1) * scale)
    );
  }

  _zeros(rows, cols) {
    return Array.from({ length: rows }, () => Array(cols).fill(0));
  }

  // ---------------------------
  // Basic matrix utilities
  // ---------------------------
  static _dot(a, b) {
    // a: [r][k], b: [k][c] -> [r][c]
    const r = a.length, k = a[0].length, c = b[0].length;
    const out = Array.from({ length: r }, () => Array(c).fill(0));
    for (let i = 0; i < r; i++) {
      for (let j = 0; j < c; j++) {
        let s = 0;
        for (let t = 0; t < k; t++) s += a[i][t] * b[t][j];
        out[i][j] = s;
      }
    }
    return out;
  }

  static _add(a, b) {
    return a.map((row, i) => row.map((v, j) => v + b[i][j]));
  }

  static _transpose(m) {
    return m[0].map((_, i) => m.map(r => r[i]));
  }

  static _mapMatrix(m, fn) {
    return m.map(r => r.map(fn));
  }

  // ---------------------------
  // Activation functions + derivatives
  // NOTE: derivative functions expect PRE-ACTIVATION z for relu,
  // and POST-ACTIVATION y for sigmoid/tanh (so we pass appropriate values).
  // ---------------------------
  static _activateMatrix(z, func) {
    if (func === 'sigmoid') return NN._mapMatrix(z, v => 1 / (1 + Math.exp(-v)));
    if (func === 'tanh') return NN._mapMatrix(z, v => Math.tanh(v));
    if (func === 'relu') return NN._mapMatrix(z, v => (v > 0 ? v : 0));
    if (func === 'linear') return z.map(r => r.slice());
    throw new Error('Unknown activation: ' + func);
  }

  static _dActivation(zOrY, func, expect = 'auto') {
    // expect: 'z' if we pass raw pre-activation z
    //         'y' if we pass post-activation y
    // For sigmoid/tanh we will call with y (post-activation), for relu call with z.
    if (func === 'sigmoid') {
      // zOrY is y
      return NN._mapMatrix(zOrY, y => y * (1 - y));
    }
    if (func === 'tanh') {
      // zOrY is y
      return NN._mapMatrix(zOrY, y => 1 - y * y);
    }
    if (func === 'relu') {
      // zOrY is z (pre-activation)
      return NN._mapMatrix(zOrY, z => (z > 0 ? 1 : 0));
    }
    if (func === 'linear') {
      return NN._mapMatrix(zOrY, _ => 1);
    }
    throw new Error('Unknown activation: ' + func);
  }

  // ---------------------------
  // Forward pass
  // returns object:
  // {
  //   zs: [ [outNodes][1] pre-activation ],
  //   activations: [ [nodes][1] post-activation ]  (first element is input column)
  // }
  // ---------------------------
  forward(inputArr) {
    // validate numeric input
    if (!Array.isArray(inputArr)) throw new Error('input must be array');
    if (inputArr.length !== this.layers[0].nodes) {
      throw new Error(`input length ${inputArr.length} !== expected ${this.layers[0].nodes}`);
    }
    const x = inputArr.map(v => {
      const n = Number(v);
      if (!Number.isFinite(n)) throw new Error('input contains non-finite number');
      return [n];
    });

    const activations = [x];
    const zs = []; // pre-activations per layer (excluding input)

    let a = x;
    for (let l = 0; l < this.weights.length; l++) {
      const w = this.weights[l]; // [out][in]
      const b = this.biases[l];  // [out][1]
      const z = NN._add(NN._dot(w, a), b); // [out][1]
      zs.push(z);
      const actFn = this.activations[l];
      a = NN._activateMatrix(z, actFn);
      activations.push(a);
    }
    return { zs, activations };
  }

  // ---------------------------
  // run: convenience wrapper to get final output as flat array
  // ---------------------------
  run(inputArr) {
    const { activations } = this.forward(inputArr);
    return activations[activations.length - 1].map(r => r[0]);
  }

  // ---------------------------
  // Memory helpers (deduplicate)
  // ---------------------------
  _isSameSample(a, b) {
    try {
      return JSON.stringify(a.input) === JSON.stringify(b.input)
          && JSON.stringify(a.output) === JSON.stringify(b.output);
    } catch {
      return false;
    }
  }

  _addUnique(sample) {
    if (!this.memory.some(m => this._isSameSample(m, sample))) {
      this.memory.push(sample);
    }
  }

  loadMemory(memoryJson) {
    const data = typeof memoryJson === 'string' ? JSON.parse(memoryJson) : memoryJson;
    if (!Array.isArray(data)) throw new Error('Memory JSON must be array of samples');
    for (const s of data) {
      if (!Array.isArray(s.input) || !Array.isArray(s.output)) continue;
      this._addUnique({ input: s.input.slice(), output: s.output.slice() });
    }
  }

  // ---------------------------
  // Train: batch gradient descent with optional minibatch behavior,
  // gradient clipping, shape checks, and NaN safeguards.
  // batch: array of samples (each {input:[], output:[]})
  // options: { iterations, errorThreshold, batchSize }
  // If batch is passed, samples are merged (unique) into memory.
  // ---------------------------
  train(batch = null, { iterations = 1000, errorThreshold = 1e-3, batchSize = null } = {}) {
    if (batch) {
      if (!Array.isArray(batch)) throw new Error('batch must be an array of samples');
      for (const s of batch) {
        if (!Array.isArray(s.input) || !Array.isArray(s.output)) continue;
        // shape check
        if (s.input.length !== this.layers[0].nodes || s.output.length !== this.layers[this.layers.length - 1].nodes) {
          throw new Error('sample shape mismatch: input or output size does not match network');
        }
        // numeric check
        s.input.forEach(v => {
          if (!Number.isFinite(Number(v))) throw new Error('training data contains non-finite input');
        });
        s.output.forEach(v => {
          if (!Number.isFinite(Number(v))) throw new Error('training data contains non-finite output');
        });
        this._addUnique({ input: s.input.map(Number), output: s.output.map(Number) });
      }
    }

    if (this.memory.length === 0) return;

    const N = this.memory.length;
    const useBatchSize = (batchSize && Number.isInteger(batchSize) && batchSize > 0) ? Math.min(batchSize, N) : N;

    for (let iter = 0; iter < iterations; iter++) {
      let totalError = 0;

      // initialize deltas
      const weightDeltas = this.weights.map(w => w.map(row => row.map(_ => 0)));
      const biasDeltas = this.biases.map(b => b.map(_ => [0]));

      // For simplicity, we'll do full pass or mini-batches sampled sequentially
      for (let bi = 0; bi < N; bi += useBatchSize) {
        const batchSamples = this.memory.slice(bi, bi + useBatchSize);

        for (const sample of batchSamples) {
          // forward
          const { zs, activations } = this.forward(sample.input);

          // compute output error (post-activation)
          const aL = activations[activations.length - 1]; // [out][1]
          const y = sample.output.map(v => [v]);

          let error = aL.map((row, i) => [y[i][0] - row[0]]); // [out][1]
          // accumulate absolute error for reporting
          totalError += error.reduce((s, r) => s + Math.abs(r[0]), 0);

          // backpropagate
          for (let l = this.weights.length - 1; l >= 0; l--) {
            const actFn = this.activations[l];
            // For derivative:
            // - if relu: we need pre-activation z
            // - if sigmoid/tanh: we can use post-activation a (activations[l+1])
            const derivInput = (actFn === 'relu') ? zs[l] : activations[l + 1];
            const deriv = NN._dActivation(derivInput, actFn);

            // gradient = deriv .* error * learningRate
            const gradient = deriv.map((r, i) => [r[0] * error[i][0] * this.learningRate]);

            // delta = gradient * activations[l]^T
            const prevActT = NN._transpose(activations[l]);
            const delta = NN._dot(gradient, prevActT);

            // accumulate weight and bias deltas
            for (let i = 0; i < delta.length; i++) {
              for (let j = 0; j < delta[0].length; j++) {
                weightDeltas[l][i][j] += delta[i][j];
              }
            }
            for (let i = 0; i < gradient.length; i++) {
              biasDeltas[l][i][0] += gradient[i][0];
            }

            // propagate error to previous layer: weights^T * gradient
            const wT = NN._transpose(this.weights[l]);
            error = NN._dot(wT, gradient);
          }
        }
      }

      // gradient clipping (prevent explosion)
      if (this.gradientClip > 0) {
        for (let l = 0; l < weightDeltas.length; l++) {
          for (let i = 0; i < weightDeltas[l].length; i++) {
            for (let j = 0; j < weightDeltas[l][0].length; j++) {
              const v = weightDeltas[l][i][j];
              if (!Number.isFinite(v)) throw new Error('NaN/Infinite encountered in weight delta');
              if (Math.abs(v) > this.gradientClip) weightDeltas[l][i][j] = Math.sign(v) * this.gradientClip;
            }
          }
        }
        for (let l = 0; l < biasDeltas.length; l++) {
          for (let i = 0; i < biasDeltas[l].length; i++) {
            const v = biasDeltas[l][i][0];
            if (!Number.isFinite(v)) throw new Error('NaN/Infinite encountered in bias delta');
            if (Math.abs(v) > this.gradientClip) biasDeltas[l][i][0] = Math.sign(v) * this.gradientClip;
          }
        }
      }

      // apply updates (note: weightDeltas were already scaled by learningRate inside gradient)
      for (let l = 0; l < this.weights.length; l++) {
        for (let i = 0; i < this.weights[l].length; i++) {
          for (let j = 0; j < this.weights[l][0].length; j++) {
            const nv = this.weights[l][i][j] + weightDeltas[l][i][j];
            if (!Number.isFinite(nv)) throw new Error('NaN/Infinite encountered while updating weights');
            this.weights[l][i][j] = nv;
          }
        }
      }
      for (let l = 0; l < this.biases.length; l++) {
        for (let i = 0; i < this.biases[l].length; i++) {
          const nv = this.biases[l][i][0] + biasDeltas[l][i][0];
          if (!Number.isFinite(nv)) throw new Error('NaN/Infinite encountered while updating biases');
          this.biases[l][i][0] = nv;
        }
      }

      if (iter % 100 === 0) console.log(`iter ${iter} totalError ${totalError.toFixed(6)}`);
      if (totalError < errorThreshold) break;
    } // iterations
  }

  // ---------------------------
  // Serialization
  // ---------------------------
  toJSON() {
    return JSON.stringify({
      layers: this.layers,
      learningRate: this.learningRate,
      gradientClip: this.gradientClip,
      weights: this.weights,
      biases: this.biases,
      activations: this.activations,
      memory: this.memory
    }, null, 2);
  }

  static fromJSON(jsonStr) {
    const obj = typeof jsonStr === 'string' ? JSON.parse(jsonStr) : obj;
    if (!obj || !Array.isArray(obj.layers)) throw new Error('Invalid JSON for NN');
    const nn = new NN({ layers: obj.layers, learningRate: obj.learningRate || 0.01, gradientClip: obj.gradientClip || 5.0 });
    if (obj.weights) nn.weights = obj.weights;
    if (obj.biases) nn.biases = obj.biases;
    if (obj.activations) nn.activations = obj.activations;
    nn.memory = Array.isArray(obj.memory) ? obj.memory.slice() : [];
    return nn;
  }
}

module.exports = NN;
