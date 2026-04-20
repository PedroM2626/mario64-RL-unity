using UnityEngine;

namespace ParkourRL
{
    public class Checkpoint : MonoBehaviour
    {
        [SerializeField] private bool activated = false;
        [SerializeField] private Color inactiveColor = Color.gray;
        [SerializeField] private Color activeColor = Color.green;
        [SerializeField] private float activationRadius = 1f;

        private Renderer checkpointRenderer;
        private Collider checkpointCollider;

        public bool IsActivated => activated;

        void Awake()
        {
            checkpointRenderer = GetComponent<Renderer>();
            checkpointCollider = GetComponent<Collider>();

            if (checkpointCollider == null)
            {
                // Adicionar trigger collider se não existir
                SphereCollider sphere = gameObject.AddComponent<SphereCollider>();
                sphere.isTrigger = true;
                sphere.radius = activationRadius;
                checkpointCollider = sphere;
            }

            UpdateVisual();
        }

        public void Activate()
        {
            if (!activated)
            {
                activated = true;
                UpdateVisual();
            }
        }

        public void Reset()
        {
            activated = false;
            UpdateVisual();
        }

        private void UpdateVisual()
        {
            if (checkpointRenderer != null)
            {
                checkpointRenderer.material.color = activated ? activeColor : inactiveColor;
            }
        }

        void OnDrawGizmos()
        {
            Gizmos.color = activated ? activeColor : inactiveColor;
            Gizmos.DrawWireSphere(transform.position, activationRadius);
        }
    }
}
