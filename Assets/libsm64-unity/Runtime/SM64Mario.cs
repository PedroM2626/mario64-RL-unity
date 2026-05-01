using System.Linq;
using UnityEngine;

namespace LibSM64
{
    public class SM64Mario : MonoBehaviour
    {
        [SerializeField] Material material = null;

        [Tooltip("Se true, nao sobrescreve _MainTex com a textura default do Mario64.")]
        public bool useCustomTexture = false;

        [Tooltip("Quando diferente de branco, substitui as vertex colors nativas do Mario por esta cor uniforme.")]
        public Color tintColor = Color.white;

        SM64InputProvider inputProvider;

        Vector3[][] positionBuffers;
        Vector3[][] normalBuffers;
        Vector3[] lerpPositionBuffer;
        Vector3[] lerpNormalBuffer;
        Vector3[] colorBuffer;
        Color[] colorBufferColors;
        Vector2[] uvBuffer;
        int buffIndex;
        Interop.SM64MarioState[] states;

        GameObject marioRendererObject;
        Mesh marioMesh;
        uint marioId;

        void OnEnable()
        {
            SM64Context.RegisterMario( this );

            var initPos = transform.position;
            marioId = Interop.MarioCreate( new Vector3( -initPos.x, initPos.y, initPos.z ) * Interop.SCALE_FACTOR );

            inputProvider = GetComponent<SM64InputProvider>();
            if( inputProvider == null )
            {
                Debug.LogError("[SM64Mario] InputProvider não encontrado!");
                throw new System.Exception("Need to add an input provider component to Mario");
            }
            if (inputProvider.GetType().Name != "MarioInputProvider")
            {
                Debug.LogWarning($"[SM64Mario] InputProvider é {inputProvider.GetType().Name}, esperado: MarioInputProvider");
            }

            marioRendererObject = new GameObject("MARIO");
            marioRendererObject.hideFlags |= HideFlags.HideInHierarchy;
            
            var renderer = marioRendererObject.AddComponent<MeshRenderer>();
            var meshFilter = marioRendererObject.AddComponent<MeshFilter>();

            states = new Interop.SM64MarioState[2] {
                new Interop.SM64MarioState(),
                new Interop.SM64MarioState()
            };

            if (material != null)
            {
                renderer.material = material;
                if (renderer.sharedMaterial != null && !useCustomTexture)
                    renderer.sharedMaterial.SetTexture("_MainTex", Interop.marioTexture);
            }

            marioRendererObject.transform.localScale = new Vector3( -1, 1, 1 ) / Interop.SCALE_FACTOR;
            marioRendererObject.transform.localPosition = Vector3.zero;

            lerpPositionBuffer = new Vector3[3 * Interop.SM64_GEO_MAX_TRIANGLES];
            lerpNormalBuffer = new Vector3[3 * Interop.SM64_GEO_MAX_TRIANGLES];
            positionBuffers = new Vector3[][] { new Vector3[3 * Interop.SM64_GEO_MAX_TRIANGLES], new Vector3[3 * Interop.SM64_GEO_MAX_TRIANGLES] };
            normalBuffers = new Vector3[][] { new Vector3[3 * Interop.SM64_GEO_MAX_TRIANGLES], new Vector3[3 * Interop.SM64_GEO_MAX_TRIANGLES] };
            colorBuffer = new Vector3[3 * Interop.SM64_GEO_MAX_TRIANGLES];
            colorBufferColors = new Color[3 * Interop.SM64_GEO_MAX_TRIANGLES];
            uvBuffer = new Vector2[3 * Interop.SM64_GEO_MAX_TRIANGLES];

            marioMesh = new Mesh();
            marioMesh.MarkDynamic();
            marioMesh.vertices = lerpPositionBuffer;
            marioMesh.triangles = Enumerable.Range(0, 3*Interop.SM64_GEO_MAX_TRIANGLES).ToArray();
            meshFilter.sharedMesh = marioMesh;
        }

        void OnDisable()
        {
            if( marioRendererObject != null )
            {
                Destroy( marioRendererObject );
                marioRendererObject = null;
            }
            
            if( marioMesh != null )
            {
                Destroy( marioMesh );
                marioMesh = null;
            }

            if( Interop.isGlobalInit )
            {
                SM64Context.UnregisterMario( this );
                Interop.MarioDelete( marioId );
            }
        }

        public void Teleport(Vector3 newPos)
        {
            // Apaga a instância nativa velha
            if( Interop.isGlobalInit ) {
                Interop.MarioDelete(marioId);
            }
            
            // Recria a instância nativa na nova posição
            marioId = Interop.MarioCreate( new Vector3( -newPos.x, newPos.y, newPos.z ) * Interop.SCALE_FACTOR );

            // Limpa os estados de transição
            states[0] = new Interop.SM64MarioState();
            states[1] = new Interop.SM64MarioState();
            buffIndex = 0;
            
            // Atualiza a posição inicial instantaneamente visualmente
            transform.position = newPos;
        }

        public void contextFixedUpdate()
        {
            // Proteção contra null durante ciclo de vida
            if (inputProvider == null || states == null || positionBuffers == null)
                return;
                
            var inputs = new Interop.SM64MarioInputs();
            var look = inputProvider.GetCameraLookDirection();
            var joystick = inputProvider.GetJoystickAxes();
            
            // Debug: logar apenas quando há input significativo (evita flood)
            if (joystick.magnitude > 0.1f && Time.frameCount % 60 == 0)
            {
                Debug.Log($"[SM64Mario] Input ativo - Joystick: {joystick}");
            }
            look.y = 0;
            look = look.normalized;

            inputs.camLookX = -look.x;
            inputs.camLookZ = look.z;
            inputs.stickX = joystick.x;
            inputs.stickY = -joystick.y;
            inputs.buttonA = inputProvider.GetButtonHeld( SM64InputProvider.Button.Jump  ) ? (byte)1 : (byte)0;
            inputs.buttonB = inputProvider.GetButtonHeld( SM64InputProvider.Button.Kick  ) ? (byte)1 : (byte)0;
            inputs.buttonZ = inputProvider.GetButtonHeld( SM64InputProvider.Button.Stomp ) ? (byte)1 : (byte)0;

            states[buffIndex] = Interop.MarioTick( marioId, inputs, positionBuffers[buffIndex], normalBuffers[buffIndex], colorBuffer, uvBuffer );

            for( int i = 0; i < colorBuffer.Length; ++i )
            {
                if (tintColor != Color.white)
                    colorBufferColors[i] = tintColor;
                else
                    colorBufferColors[i] = new Color( colorBuffer[i].x, colorBuffer[i].y, colorBuffer[i].z, 1 );
            }

            buffIndex = 1 - buffIndex;
        }

        public void contextUpdate()
        {
            // Proteção contra null durante ciclo de vida
            if (lerpPositionBuffer == null || states == null)
                return;
                
            float t = (Time.time - Time.fixedTime) / Time.fixedDeltaTime;
            int j = 1 - buffIndex;

            for( int i = 0; i < lerpPositionBuffer.Length; ++i )
            {
                lerpPositionBuffer[i] = Vector3.LerpUnclamped( positionBuffers[buffIndex][i], positionBuffers[j][i], t );
                lerpNormalBuffer[i] = Vector3.LerpUnclamped( normalBuffers[buffIndex][i], normalBuffers[j][i], t );
            }

            transform.position = Vector3.LerpUnclamped( states[buffIndex].unityPosition, states[j].unityPosition, t );

            marioMesh.vertices = lerpPositionBuffer;
            marioMesh.normals = lerpNormalBuffer;
            
            // As atualizações de Colors e UVs ficam no Update visual para não estourar o TLS Allocator no Unity ML-Agents (TimeScale alto)
            marioMesh.colors = colorBufferColors;
            marioMesh.uv = uvBuffer;

            marioMesh.RecalculateBounds();
            // marioMesh.RecalculateTangents(); // Desabilitado para evitar vazamentos ALLOC_TEMP_MAIN (Desnecessário sem Normal Map)
        }

        void OnDrawGizmos()
        {
            if( !Application.isPlaying )
            {
                Gizmos.color = Color.red;
                Gizmos.DrawSphere( transform.position, 0.5f );
            }
        }
    }
}