using UnityEngine;

namespace ParkourRL.BackupSystem
{
    public class BackupCheckpoint : MonoBehaviour
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

        public void ResetCheckpoint()
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
