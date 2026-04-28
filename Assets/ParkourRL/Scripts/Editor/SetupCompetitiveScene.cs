using UnityEngine;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine.SceneManagement;

namespace ParkourRL.Editor
{
    public class SetupCompetitiveScene
    {
        [MenuItem("Parkour RL/Setup Competitive Scene")]
        public static void SetupScene()
        {
            string oldScenePath = "Assets/ParkourRL/Scenes/ParkourTraining_OldSystem.unity";
            string newScenePath = "Assets/ParkourRL/Scenes/CompetitiveParkour.unity";

            if (!System.IO.File.Exists(oldScenePath))
            {
                // Fallback para a cena normal se a OldSystem não estiver presente
                oldScenePath = "Assets/ParkourRL/Scenes/ParkourTraining.unity";
                if (!System.IO.File.Exists(oldScenePath))
                {
                    Debug.LogError("Nenhuma cena base de Parkour encontrada para copiar!");
                    return;
                }
            }

            // Copiar a cena fisicamente para preservar os GameObjects
            if (System.IO.File.Exists(newScenePath))
            {
                if (EditorUtility.DisplayDialog("Aviso", "Isso vai substituir completamente sua cena 'CompetitiveParkour' pelo mapa do parkour antigo para o modo simultâneo. Tem certeza?", "Sim, substituir", "Cancelar"))
                {
                    System.IO.File.Copy(oldScenePath, newScenePath, true);
                }
                else
                {
                    return;
                }
            }
            else
            {
                System.IO.File.Copy(oldScenePath, newScenePath);
            }

            AssetDatabase.Refresh();

            // Abrir a cena no Editor
            Scene newScene = EditorSceneManager.OpenScene(newScenePath);

            // Trocar o script ParkourEnvironment pelo CompetitiveParkourEnvironment
            ParkourEnvironment oldEnv = Object.FindObjectOfType<ParkourEnvironment>();
            if (oldEnv != null)
            {
                GameObject envObj = oldEnv.gameObject;
                
                // Pegar referências antes de deletar
                SerializedObject so = new SerializedObject(oldEnv);
                
                SerializedProperty spawnPointsProp = so.FindProperty("spawnPoints");
                Transform firstSpawn = null;
                if (spawnPointsProp != null && spawnPointsProp.arraySize > 0)
                {
                    firstSpawn = spawnPointsProp.GetArrayElementAtIndex(0).objectReferenceValue as Transform;
                }

                SerializedProperty goalProp = so.FindProperty("goal");
                Transform goalTransform = null;
                if (goalProp != null)
                {
                    goalTransform = goalProp.objectReferenceValue as Transform;
                }

                // Deletar o manager antigo e colocar o do modo simultâneo
                Object.DestroyImmediate(oldEnv);
                CompetitiveParkourEnvironment compEnv = envObj.AddComponent<CompetitiveParkourEnvironment>();

                // Transferir referências de spawn e goal para ele
                SerializedObject compSo = new SerializedObject(compEnv);
                
                if (firstSpawn != null)
                {
                    SerializedProperty compSpawnPointsProp = compSo.FindProperty("spawnPoints");
                    if (compSpawnPointsProp != null)
                    {
                        compSpawnPointsProp.arraySize = 1;
                        compSpawnPointsProp.GetArrayElementAtIndex(0).objectReferenceValue = firstSpawn;
                    }
                }

                if (goalTransform != null)
                {
                    SerializedProperty compGoalProp = compSo.FindProperty("goal");
                    if (compGoalProp != null)
                    {
                        compGoalProp.objectReferenceValue = goalTransform;
                    }
                }

                // Definir os parametros base
                SerializedProperty marioCountProp = compSo.FindProperty("marioCount");
                if (marioCountProp != null) marioCountProp.intValue = 3;

                compSo.ApplyModifiedProperties();
                
                Debug.Log("[Parkour RL] Cena CompetitiveParkour gerada fisicamente com sucesso! Todos os objetos estão na Hierarchy.");
            }
            else
            {
                Debug.LogWarning("Não foi possível encontrar o ambiente base para linkar o spawn, mas o cenário foi copiado!");
            }

            // Salvar a cena recém montada
            EditorSceneManager.SaveScene(newScene);
        }
    }
}
