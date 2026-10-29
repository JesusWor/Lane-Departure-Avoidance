# Manual de instalación y pruebas en Windows
## Lane Departure Avoidance (LDA)

**Objetivo:** instalar las dependencias, compilar el autómata en C++, comprobar su comunicación con Python y probar el sistema primero sin CARLA y después con el simulador.

> **Importante:** este manual está basado en la estructura y los comandos que hemos manejado. No significa que se haya compilado y validado todo en tu computadora. Si un comando falla, revisa el mensaje exacto antes de continuar.

---

## 0. Estructura del proyecto

La carpeta principal debería tener esta estructura:

```text
lda_project/
├── README.md
├── WINDOWS.md
├── automaton/
│   ├── CMakeLists.txt
│   ├── include/
│   │   ├── codec.hpp
│   │   ├── fsm.hpp
│   │   ├── lane_change.hpp
│   │   ├── lane_keeping.hpp
│   │   └── types.hpp
│   ├── src/
│   │   └── main.cpp
│   └── tests/
│       └── test_logic.cpp
├── python/
│   ├── link.py
│   ├── perception.py
│   ├── mock_sim.py
│   ├── run_carla.py
│   └── requirements.txt
└── YOLOPv2/
    └── ... archivos originales del repositorio ...
```

Los archivos `mock_sim.py` y `run_carla.py` deben estar en `python/`, no dentro de `YOLOPv2/`.

Si alguno de estos archivos no existe en tu proyecto, no lo crees vacío para completar la estructura: primero confirma si realmente lo necesita tu versión del código.

---

## 1. Instalar las herramientas

Este paso se realiza una sola vez.

### 1.1. Visual Studio y CMake

Instala **Visual Studio 2022 Community** o **Visual Studio 2022 Build Tools**, incluyendo la carga de trabajo:

- Desarrollo de escritorio con C++.
- MSVC para x64.
- Windows SDK.

Instala también Git y CMake. Si `winget` está disponible, abre PowerShell y ejecuta:

```powershell
winget install Kitware.CMake
winget install Git.Git
```

Cierra y vuelve a abrir PowerShell después de la instalación.

Verifica que estén disponibles:

```powershell
cmake --version
git --version
```

Si `cmake` no se reconoce, revisa que esté agregado al `PATH` o abre una terminal nueva.

### 1.2. Instalar vcpkg

Abre PowerShell y ejecuta:

```powershell
git clone https://github.com/microsoft/vcpkg C:\vcpkg
C:\vcpkg\bootstrap-vcpkg.bat
```

Instala las bibliotecas utilizadas por el autómata:

```powershell
C:\vcpkg\vcpkg install zeromq cppzmq nlohmann-json --triplet x64-windows
```

Estas dependencias sirven para la comunicación mediante ZeroMQ y el manejo de mensajes JSON.

Si ya tienes vcpkg instalado en otra carpeta, utiliza esa ruta en los comandos siguientes en lugar de instalarlo nuevamente.

---

## 2. Compilar el autómata de C++

Abre PowerShell y entra en la carpeta `automaton`:

```powershell
cd "RUTA\lda_project\automaton"
```

Reemplaza `RUTA` por la ubicación real de tu proyecto.

Configura CMake con vcpkg:

```powershell
cmake -S . -B build `
  -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake `
  -DVCPKG_TARGET_TRIPLET=x64-windows
```

Compila en modo Release:

```powershell
cmake --build build --config Release
```

### 2.1. Comprobar los ejecutables

Con el generador de Visual Studio, normalmente se generan en `build\Release`. Comprueba qué archivos se crearon:

```powershell
Get-ChildItem .\build -Recurse -Filter *.exe
```

Deberías localizar el ejecutable principal, `lda_automaton.exe`, y el ejecutable de pruebas, `test_logic.exe`, si ambos están definidos en tu `CMakeLists.txt`.

Ejecuta las pruebas unitarias desde la ubicación real del ejecutable. Por ejemplo, si está en `build\Release`:

```powershell
.\build\Release\test_logic.exe
```

Si está directamente en `build`:

```powershell
.\build\test_logic.exe
```

**Resultado esperado:** que las pruebas terminen correctamente y no reporten fallos. El número de pruebas depende de la versión de `test_logic.cpp`; no asumas que necesariamente son 25.

### 2.2. Si falta una DLL

Si Windows informa que falta una DLL de ZeroMQ, revisa si se encuentra en:

```text
C:\vcpkg\installed\x64-windows\bin\
```

Si es necesario, copia la DLL correspondiente junto al ejecutable que la necesita. Para un ejecutable en `build\Release`, puedes utilizar:

```powershell
Copy-Item "C:\vcpkg\installed\x64-windows\bin\libzmq*.dll" `
  ".\build\Release\"
```

Ejecuta el comando desde `automaton/`. Si no encuentra archivos, revisa los nombres reales de las DLL disponibles antes de copiarlas.

---

## 3. Preparar el entorno de Python

Usa Python 3.10 de 64 bits para mantener la compatibilidad con CARLA 0.9.15. Esa versión de CARLA publica una distribución de su API para CPython 3.10 en Windows x64. [Referencia de CARLA en PyPI](https://pypi.org/project/carla/0.9.15/).

Entra a la carpeta `python`:

```powershell
cd "RUTA\lda_project\python"
```

Crea un entorno virtual:

```powershell
py -3.10 -m venv .venv
```

Actívalo en PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Si estás utilizando CMD en lugar de PowerShell, el comando de activación es:

```bat
.venv\Scripts\activate.bat
```

Actualiza `pip` e instala las dependencias del proyecto:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Comprueba que el entorno está usando Python 3.10:

```powershell
python --version
python -m pip --version
```

Instala la API de CARLA solamente si vas a ejecutar las pruebas con el simulador:

```powershell
python -m pip install carla==0.9.15
```

Comprueba la instalación:

```powershell
python -c "import carla; print('CARLA API instalada:', carla.__file__)"
```

Si falla la instalación, verifica que el entorno virtual siga activo y que la versión de Python coincida con la distribución disponible.

---

## 4. Primera etapa: probar sin CARLA

Esta etapa es la más importante. Permite comprobar la lógica del autómata y la comunicación entre C++ y Python sin agregar todavía los problemas propios del simulador o de la visión artificial.

### 4.1. Entender los cuatro escenarios

| Escenario | Qué simula | Qué deberías comprobar |
|---|---|---|
| `free` | No hay obstáculos que impidan el cambio | El sistema puede autorizar el cambio si las demás condiciones son válidas |
| `blocked` | Hay un obstáculo en el carril objetivo | El sistema rechaza el cambio |
| `rear` | Hay un vehículo que se aproxima desde atrás | El validador considera el riesgo y puede rechazar el cambio |
| `stale` | Se interrumpe temporalmente la percepción | El autómata detecta datos desactualizados y entra en `DEGRADED` |

Los resultados concretos dependen de los umbrales y las reglas implementadas en `lane_change.hpp` y `fsm.hpp`.

### 4.2. Abrir la primera terminal: C++

Abre PowerShell en `automaton/` y ejecuta el programa. Si el ejecutable está en `build\Release`:

```powershell
.\build\Release\lda_automaton.exe -v --ff 0
```

Si tu ejecutable está en `build/`, ajusta la ruta.

Si aparece un error sobre `-v` o `--ff`, consulta las opciones que admite tu versión del programa. No todos los ejecutables reconocen los mismos argumentos.

Deja esta terminal abierta. El autómata debe seguir ejecutándose mientras Python envía datos.

### 4.3. Abrir la segunda terminal: Python

Abre otra terminal PowerShell, entra en `python/` y activa el entorno virtual:

```powershell
cd "RUTA\lda_project\python"
.\.venv\Scripts\Activate.ps1
```

Ejecuta el primer escenario:

```powershell
python mock_sim.py --scenario free
```

Al finalizar, revisa que las verificaciones aparezcan como `[PASS]`.

Después prueba los escenarios restantes:

```powershell
python mock_sim.py --scenario blocked
python mock_sim.py --scenario rear
python mock_sim.py --scenario stale
```

Es preferible ejecutar un escenario a la vez y esperar a que termine antes de iniciar el siguiente.

### 4.4. Qué revisar en los resultados

No te limites a que todos los escenarios terminen con `[PASS]`. Comprueba también:

- Que el vehículo simulado corrija su desviación cuando corresponda.
- Que el resultado del cambio de carril responda a los obstáculos.
- Que la latencia reportada tenga sentido y no haya tiempos de espera agotados.
- Que el escenario `stale` detecte la interrupción de la percepción.
- Que al restablecerse la percepción, el sistema vuelva a evaluar los datos nuevos.

**Importante sobre `stale`:** la prueba existente puede verificar que el autómata entra en `DEGRADED`, pero eso no demuestra automáticamente que una nueva solicitud de cambio de carril realizada durante la interrupción sea rechazada. Para demostrarlo, la prueba debe enviar una solicitud durante ese estado y comprobar que la respuesta sea `false`.

Cambiar el valor esperado de una prueba para que coincida con el resultado observado no modifica la lógica de C++; solamente modifica lo que la prueba considera correcto. Por eso es importante verificar el comportamiento que realmente quieres validar.

### 4.5. Puertos de comunicación

En la configuración actual:

- Python publica datos hacia `tcp://127.0.0.1:5555`.
- C++ recibe esos datos.
- C++ publica sus decisiones en `tcp://127.0.0.1:5556`.
- Python recibe las decisiones.

No ejecutes dos instancias del autómata que intenten usar los mismos puertos simultáneamente.

Si aparece `TIMEOUT`, revisa primero que el autómata esté funcionando y que ambos programas utilicen las mismas direcciones y puertos. Si sospechas que quedó un proceso anterior, ciérralo antes de volver a iniciar la prueba.

---

## 5. Segunda etapa: preparar CARLA

Realiza esta etapa únicamente después de que las pruebas sin CARLA funcionen.

### 5.1. Instalar e iniciar CARLA

Descarga el paquete de CARLA 0.9.15 para Windows desde sus [releases oficiales](https://github.com/carla-simulator/carla/releases).

Extrae el paquete y localiza `CarlaUE4.exe`.

Inicia el simulador y espera a que termine de cargar el mapa. Si la carga falla o la computadora se queda sin recursos, revisa primero la configuración gráfica y los requisitos del paquete.

CARLA utiliza normalmente los puertos TCP 2000 y 2001 para la conexión y la comunicación con el cliente. Asegúrate de que no estén bloqueados ni ocupados. [Documentación de CARLA 0.9.15](https://carla.readthedocs.io/en/0.9.15/start_quickstart/).

### 5.2. Iniciar el autómata

En una terminal independiente, entra en `automaton/` e inicia el ejecutable C++ como en la etapa anterior.

### 5.3. Ejecutar la integración con CARLA

En otra terminal, activa el entorno virtual y consulta primero las opciones disponibles:

```powershell
python run_carla.py --help
```

Si el script admite los argumentos indicados en esta guía, ejecuta:

```powershell
python run_carla.py --mode gt --request LEFT --request-at 8
```

El modo `gt` debe utilizar los datos que el propio script obtiene del simulador. Confirma en el código qué información utiliza antes de interpretar los resultados.

No pases a YOLOPv2 hasta que esta prueba de integración funcione.

---

## 6. Tercera etapa: YOLOPv2

YOLOPv2 es una etapa adicional de percepción basada en visión artificial. No es necesaria para comprobar la lógica del autómata.

Clona el repositorio original en la raíz del proyecto:

```powershell
cd "RUTA\lda_project"
git clone https://github.com/CAIC-AD/YOLOPv2 YOLOPv2
```

Sigue las instrucciones de instalación y descarga de pesos publicadas por el proyecto. Comprueba que los archivos de pesos estén en la ruta que espera tu código.

### Consideraciones sobre PyTorch

El repositorio de YOLOPv2 contiene un archivo de dependencias antiguo que fija versiones específicas de PyTorch y torchvision. No instales automáticamente todo ese archivo en un entorno moderno sin comprobar la compatibilidad: podría reemplazar paquetes que ya funcionan o provocar conflictos. [Dependencias originales de YOLOPv2](https://github.com/CAIC-AD/YOLOPv2/blob/main/requirements.txt).

Instala una combinación compatible con tu versión de Python y con tu equipo, y después verifica que puedas importar los paquetes requeridos.

Si tu script admite el modo YOLOPv2, ejecuta:

```powershell
python run_carla.py --help
```

Y, si aparece `yolop` entre los modos disponibles:

```powershell
python run_carla.py --mode yolop
```

La integración solo estará completa cuando la salida de la percepción llegue correctamente al autómata. Tener YOLOPv2 instalado no garantiza por sí solo que la comunicación y los datos estén conectados.

---

## 7. Solución de problemas

| Síntoma | Qué revisar |
|---|---|
| `cmake` no se reconoce | Reabrir PowerShell y comprobar el `PATH` |
| No se encuentra `nlohmann/json.hpp` | Verificar que vcpkg instaló `nlohmann-json` y que CMake utiliza el toolchain correcto |
| Falta una DLL al ejecutar C++ | Revisar las DLL instaladas por vcpkg y la carpeta del ejecutable |
| No existe `build\Release` | Buscar los `.exe` con `Get-ChildItem`; la ruta depende del generador de CMake |
| `TIMEOUT` entre Python y C++ | Comprobar procesos, puertos, direcciones y que ambas aplicaciones estén activas |
| `No matching distribution found for carla` | Comprobar Python, arquitectura y versión de CARLA |
| No conecta con CARLA | Esperar a que el servidor termine de cargar y revisar los puertos 2000 y 2001 |
| El coche gira en dirección contraria | Revisar el signo del error lateral y la conversión entre corrección y dirección |
| Falla la prueba `stale` | Comprobar el tiempo de espera, el umbral de datos antiguos y qué se está validando exactamente |
| Falla `import torch` o `import torchvision` | Verificar las versiones instaladas y su compatibilidad con Python y el código de YOLOPv2 |

---

## 8. Orden recomendado de validación

Sigue este orden y conserva los resultados de cada etapa:

1. Compilar C++ sin errores.
2. Ejecutar las pruebas unitarias del autómata.
3. Ejecutar `free` y verificar la autorización del cambio de carril.
4. Ejecutar `blocked` y `rear` y verificar los rechazos esperados.
5. Ejecutar `stale` y comprobar la entrada a `DEGRADED` y el rechazo de solicitudes durante la pérdida de percepción.
6. Ejecutar la integración con CARLA en modo `gt`.
7. Conectar YOLOPv2 y comprobar que la percepción se transmite correctamente.
8. Repetir las pruebas y comparar latencias, estados y resultados.

## 9. Limitaciones que debes tener presentes

- `mock_sim.py` genera datos simulados; no demuestra que una cámara real esté detectando las líneas del carril.
- El modo `gt` depende de cómo `run_carla.py` obtiene la información del simulador.
- Una respuesta `true` indica que el validador permite el cambio; no significa que el vehículo ya haya ejecutado la maniobra.
- Entrar en `DEGRADED` y asignar `steer_rad = 0` no garantiza que un vehículo real permanezca en su carril.
- Las latencias medidas en una simulación local no garantizan los mismos tiempos en un vehículo real.

**Criterio final:** considera que una etapa está validada solo cuando tienes evidencia de que se comporta como esperas. No confundas una prueba que termina correctamente con una validación completa del sistema de asistencia.
