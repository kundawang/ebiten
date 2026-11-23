// Copyright 2026 The Ebiten Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package gamepadprofile

// Hardware input limits used when validating a profile. They follow the fixed button count the
// engine exposes and the typical evdev axis and hat bounds, and reject obviously impossible indices
// without depending on any platform backend.
const (
	rawButtonCount = 32
	rawAxisCount   = 32
	rawHatCount    = 4
)
