# Lecture 1

Example - Pendulum
---
1. Model: F = m⋅a (Newtons second law). Assume m= 1 and then do some math and end up with the equation d²θ/dt² + g/l ⋅ sin(θ) = 0 (where θ is angle, g is gravity and l is length of line)
2. Solution via Jacobi elliptic integrals (whatever this means). Expand to two variables: Write dθ/dt=v and dv/dt = -g/l ⋅ sin(θ). Then get the energy _H = 1/2 ⋅ v² - g/l ⋅ cos(θ)_
3. Use Eulers Method. Let u = (θ, v), then u(h) = u(0) + h ⋅ du/dt(0) + O(h²). Praise Satan and then you're done.