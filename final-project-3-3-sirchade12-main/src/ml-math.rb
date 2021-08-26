def mean x
  # BEGIN YOUR CODE
  total = 0.0
  for i in x
    total = total + i
  end
  return total/x.size
  # END YOUR CODE
end

def stdev x
  # BEGIN YOUR CODE
  total = 0.0
  mean = mean(x)
  for i in x
    total = total + (i - mean).abs()**2
  end
  total = total/(x.size() - 1)
  total = Math.sqrt(total)
  return total
  # END YOUR CODE
end

# Add any code you want here, or remove this section. All code must be your own

# BEGIN YOUR CODE
def dot x, w
  # BEGIN YOUR CODE
  dot = 0
  for i in x
    if w.has_key?(i[0]) and x[i[0]].is_a?Numeric and  w[i[0]].is_a?Numeric and x.has_key?(i[0])
      dot = dot + w[i[0]] * x[i[0]]
    end
  end
  return dot
  # END YOUR CODE
end

def norm w
  # BEGIN YOUR CODE
  return Math.sqrt(dot(w,w,))
  # END YOUR CODE
end
# END YOUR CODE
