--[[
Implementation of
   Shim et al.,
   "SVD-Softmax: Fast Softmax Approximation on Large Vocabulary Neural Networks",
   NIPS 2017.

The module below replaces the nn.Linear module before the softmax module (nn.Softmax or nn.LogSoftmax)

nn.SVDLinear, which is a subclass of nn.Linear, implements two additional methods doSVD() and setSVDParam(W,N).
  doSVD():            performs SVD on self.weight and prepares parameters needed for SVD-based linear.
  setSVDParam(W,N):   W and N are hyperparameters that specifies the number of singular values to use (W)
                      and the number of dimensions to fully update (N).

nn.SVDLinear overloads the function updateOutput() of nn.Linear.
During training (self.train == true) or if SVD weights are not properly set up, superclass method is called instead.
--]]

local SVDLinear, parent = torch.class('nn.SVDLinear', 'nn.Linear')

function SVDLinear:load(nnLinear)
   self.weight = nnLinear.weight:clone()
   self.gradWeight = nnLinear.gradWeight:clone()
   if nnLinear.bias then
      self.bias = nnLinear.bias:clone()
      self.gradBias = nnLinear.gradBias:clone()
   end
end

--[[
  self.weight:   V x D
  self.weight_U: V x D
  self.weight_S: D
  self.weight_V: D x D
  self.weight_B: V x D

  To reconstruct weight: U * torch.diag(S) * V:t()
--]]
function SVDLinear:doSVD()
  assert(self.weight:nDimension() == 2)

  self.weight_U, self.weight_S, self.weight_V = torch.svd(self.weight)
  self.weight_B = self.weight_U * torch.diag(self.weight_S)   -- V x D
end

function SVDLinear:setSVDParam(W, N)
  if W == nil or N == nil then
    W = math.ceil(self.weight:size(2)/5)
    N = math.ceil(self.weight:size(1)/10)
  end

  assert(type(W) == "number", "SVDLinear:setSVDParam(): W is not a number")
  assert(type(N) == "number", "SVDLinear:setSVDParam(): N is not a number")
  assert(W < self.weight:size(2), "SVDLinear:setSVDParam(): W is equal to or greater than input dimension")
  assert(N < self.weight:size(1), "SVDLinear:setSVDParam(): N is equal to or greater than output dimension")

  self.svd_param_W = W
  self.svd_param_N = N
  if self.weight_B == nil then doSVD() end
  self.weight_B_W = self.weight_B[{{},{1,self.svd_param_W}}]:contiguous()
end

--[[
Cn:   N (xB)
z:    V (xB)
B:    V x D
h:    D (xB)
bias: V
--]]
function SVDLinear._updateFullView(indices, z, B, h, bias)
  if indices:nDimension() == 1 then
    for i = 1,indices:size(1) do
      local vIdx = indices[i]
      z[vIdx] = B[vIdx] * h  + (bias ~= nil and bias[vIdx] or 0)
    end
  elseif indices:nDimension() == 2 then
    for bIdx = 1,indices:size(2) do
      for nIdx = 1,indices:size(1) do
        local vIdx = indices[nIdx][bIdx]
        z[vIdx][bIdx] = B[vIdx] * h[{{},bIdx}] + (bias ~= nil and bias[vIdx] or 0)
      end
    end
  end
end

--[[
  input: B x D
  output: B x V

  self.weight: V x D
  self.bias: nil or V
--]]
function SVDLinear:updateOutput(input)

  if self.train then

    parent.updateOutput(self, input)

  else

    if self.svd_param_W == nil or self.svd_param_N == nil or self.weight_B_W == nil then
      self:setSVDParam()
    end

    -- 1. compute preview outputs with W dimensions
    local z_tilda, h_tilda
    z_tilda = self.output
    if self.h_tilda_cache == nil then self.h_tilda_cache = torch.Tensor() end
    h_tilda = self.h_tilda_cache

    if input:nDimension() == 1 then
      h_tilda:resize(self.weight_V:size(1))
      h_tilda:mv(self.weight_V:t(), input)                -- D
      z_tilda:resize(self.weight:size(1))
      z_tilda:mv(self.weight_B_W,
                 h_tilda:sub(1,self.svd_param_W))         -- V
      if self.bias then
        z_tilda:add(self.bias)
      end
    elseif input:nDimension() == 2 then
      h_tilda:resize(self.weight_V:size(1), input:size(1))-- D x B
      h_tilda:mm(self.weight_V:t(), input:t())            -- D x B
      z_tilda:resize(self.weight:size(1), input:size(1))
      z_tilda:mm(self.weight_B_W,
                 h_tilda:sub(1,self.svd_param_W))         -- V x B
      if self.bias then
        z_tilda:add(nn.utils.addSingletonDimension(self.bias,2):expand(self.bias:size(1), input:size(1)))
      end
    else
      assert(false, "nn.SVDLinear:updateOutput(): input should have either 1 or 2 dimensions")
    end

    -- 2. select N words of largest preview outputs
    local _, Cn = torch.topk(z_tilda, self.svd_param_N, 1, true) -- retrieve top-k largest elements (_) and their indices (Cn)
    -- Cn: N (x B)

    -- 3. update selected words by full-view vector multiplication
    if h_tilda.THNN.SVDLinear_updateFullView then

      if not Cn:isContiguous() then Cn = Cn:contiguous() end
      
      h_tilda.THNN.SVDLinear_updateFullView(Cn:cdata(),
                                            z_tilda:cdata(),
                                            self.weight_B:cdata(),
                                            h_tilda:cdata(),
                                            self.bias:cdata())
    else
      self._updateFullView(Cn, z_tilda, self.weight_B, h_tilda, self.bias)
    end

    if z_tilda:nDimension() == 2 then
      self.output = z_tilda:t()
    end
  end

  return self.output
end
