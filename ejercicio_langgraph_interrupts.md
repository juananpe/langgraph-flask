# Ejercicio: Aplicación Web con LangGraph Interrupts para Aprobación de Tweets

## Objetivo
Implementa una aplicación web Flask (pide a Claude 4 Sonnet que implemente) un flujo de trabajo con  "human-in-the-loop" usando LangGraph `interrupts` para la generación y aprobación de tweets.

## Descripción del Proyecto
Crear un sistema de aprobación de tweets donde un agente de IA genera contenido para redes sociales, pero requiere aprobación humana antes de publicar. El sistema debe permitir al usuario revisar, aprobar, rechazar o solicitar mejoras al contenido generado.

## Requisitos Técnicos

### Dependencias Requeridas
```
langgraph
langchain[openai]
python-dotenv
Flask
```

### Variables de Entorno
- `OPENAI_API_KEY`: Clave de API de OpenAI

## Funcionalidades a Implementar

### 1. Backend (Flask)

#### Estructura del Agente LangGraph
- **Estado**: Debe manejar una lista de mensajes
- **Nodos**:
  - `generate`: Genera el tweet inicial y pausa con `interrupt()`
  - `post`: Publica el tweet aprobado
  - `feedback`: Recolecta retroalimentación del usuario
  - `regenerate`: Mejora el tweet basado en la retroalimentación
- **Flujo**: Debe permitir ciclos de retroalimentación hasta obtener aprobación

#### Endpoints de la API
1. **`GET /`**: Sirve la interfaz web
2. **`POST /generate`**: 
   - Recibe un tema para el tweet
   - Inicia el agente LangGraph
   - Maneja la excepción del interrupt
   - Extrae el contenido del tweet del estado interrumpido
   - Retorna: `{'status': 'interrupted', 'thread_id': '...', 'post_content': '...'}`

3. **`POST /resume`**:
   - Recibe `thread_id` y `decision` (approve/reject/edit)
   - Usa `Command(resume=...)` para continuar el agente
   - Maneja diferentes rutas según la decisión

4. **`POST /feedback`**:
   - Recibe `thread_id` y `feedback`
   - Reanuda el agente con la retroalimentación
   - Maneja el nuevo interrupt con el tweet mejorado

### 2. Frontend (HTML/JavaScript)

#### Interfaz de Usuario
- **Formulario de entrada**: Campo para ingresar el tema del tweet
- **Área de visualización**: Muestra el tweet generado
- **Botones de acción**: 
  - ✅ Aprobar y Publicar
  - ❌ Rechazar
  - ✏️ Solicitar Cambios
- **Formulario de retroalimentación**: Área de texto para mejoras
- **Indicadores de estado**: Mensajes informativos para el usuario

#### Funcionalidades JavaScript
- Manejo de formularios con `fetch()` API
- Gestión del `thread_id` para mantener la sesión
- Actualización dinámica de la interfaz según las respuestas del servidor
- Manejo de errores y estados de carga

## Aspectos Técnicos Importantes

### Manejo de Interrupts
```python
# En el nodo generate
interrupt({
    "type": "approval_request",
    "message": "¿Desea publicar este tweet?",
    "post_content": post_content,
    "options": ["approve", "edit", "reject"]
})
```

### Extracción de Datos del Interrupt
```python
# Acceso correcto a los datos del interrupt
interrupt_obj = state.tasks[0].interrupts[0]
interrupt_data = interrupt_obj.value  # El diccionario que pasamos a interrupt()
post_content = interrupt_data.get('post_content', 'Sin contenido')
```

### Configuración del Checkpointer
```python
from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()
graph = graph.compile(checkpointer=checkpointer)
```

### Manejo de Thread IDs
```python
config = {"configurable": {"thread_id": str(uuid.uuid4())}}
```

## Criterios de Evaluación

### Funcionalidad (40%)
- ✅ El agente genera tweets correctamente
- ✅ Los interrupts funcionan y pausan la ejecución
- ✅ El sistema puede reanudar desde el punto de interrupción
- ✅ El ciclo de retroalimentación funciona completamente

### Implementación Técnica (30%)
- ✅ Uso correcto de `interrupt()` y `Command(resume=...)`
- ✅ Manejo apropiado de excepciones en interrupts
- ✅ Configuración correcta del checkpointer
- ✅ Gestión adecuada de thread IDs

### Interfaz de Usuario (20%)
- ✅ Interfaz intuitiva y responsive
- ✅ Manejo correcto de estados (carga, error, éxito)
- ✅ Flujo de usuario claro y lógico
- ✅ Retroalimentación visual apropiada

### Código y Documentación (10%)
- ✅ Código limpio y bien estructurado
- ✅ Comentarios explicativos en partes complejas
- ✅ Manejo de errores robusto
- ✅ Separación clara entre frontend y backend

## Entregables

1. **Código fuente completo** de la aplicación Flask
2. **Archivo de requisitos** (`requirements.txt`)
3. **Documentación** explicando:
   - Cómo configurar y ejecutar la aplicación
   - Flujo de datos entre frontend y backend
   - Explicación del uso de LangGraph interrupts
4. **Demo en vivo** mostrando todas las funcionalidades

## Recursos de Apoyo

- [Documentación oficial de LangGraph Interrupts](https://langchain-ai.lang.chat/langgraph/agents/human-in-the-loop/)
- [Ejemplos de Command(resume=...)](https://langchain-ai.lang.chat/langgraph/reference/types/)
- [Flask Documentation](https://flask.palletsprojects.com/)

## Tiempo Estimado
**4-6 horas** para estudiantes con conocimientos básicos de Python y Flask.

## Extensiones Opcionales (Puntos Extra)

1. **Persistencia**: Usar PostgreSQL o Redis en lugar de `InMemorySaver`
2. **Autenticación**: Sistema de usuarios para múltiples sesiones
3. **Historial**: Guardar y mostrar tweets anteriores
4. **Plantillas**: Diferentes tipos de tweets (promocional, informativo, etc.)
5. **Análisis**: Métricas sobre aprobaciones/rechazos
6. **API REST**: Documentación completa con Swagger/OpenAPI

---

**¡Buena suerte con el desarrollo! Este ejercicio les dará experiencia práctica con patrones avanzados de IA y desarrollo web.**
