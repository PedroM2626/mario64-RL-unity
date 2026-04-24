using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Spinner : MonoBehaviour
{
    LibSM64.SM64DynamicTerrain terrain;

    void Start()
    {
        terrain = GetComponent<LibSM64.SM64DynamicTerrain>();
        if (terrain == null)
        {
            Debug.Log("[Spinner] SM64DynamicTerrain nao encontrado neste objeto. Spinner desabilitado para este objeto.");
            enabled = false;
        }
    }

    void FixedUpdate()
    {
        if (terrain == null)
            return;

        terrain.SetRotation( Quaternion.AngleAxis( -20 * Time.time, Vector3.up ));
    }
}
