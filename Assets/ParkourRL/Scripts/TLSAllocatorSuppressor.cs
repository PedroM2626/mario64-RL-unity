using UnityEngine;

namespace ParkourRL
{
    /// <summary>
    /// Suprime os warnings de "TLS Allocator ALLOC_TEMP_TLS" que sao gerados
    /// pela DLL nativa sm64.dll. Esses warnings nao representam vazamentos reais
    /// de memoria - sao alocacoes estaticas do motor C do SM64 que a Unity nao
    /// consegue rastrear porque pertencem ao codigo nativo (unmanaged).
    /// 
    /// Os enderecos reportados sao sempre os mesmos (ex: 0000022A211114A0),
    /// confirmando que sao buffers fixos alocados uma unica vez na inicializacao.
    /// </summary>
    public static class TLSAllocatorSuppressor
    {
        // Filtro customizado que intercepta logs antes de chegarem ao Console
        private class SuppressiveLogHandler : ILogHandler
        {
            private readonly ILogHandler defaultHandler;

            public SuppressiveLogHandler(ILogHandler defaultHandler)
            {
                this.defaultHandler = defaultHandler;
            }

            public void LogFormat(LogType logType, Object context, string format, params object[] args)
            {
                // Somente filtrar warnings e erros do TLS Allocator nativo
                if (logType == LogType.Warning || logType == LogType.Error || logType == LogType.Log)
                {
                    string message = (args != null && args.Length > 0) 
                        ? string.Format(format, args) 
                        : format;
                    
                    if (IsTLSAllocatorNoise(message))
                        return; // Descarta silenciosamente
                }

                defaultHandler.LogFormat(logType, context, format, args);
            }

            public void LogException(System.Exception exception, Object context)
            {
                defaultHandler.LogException(exception, context);
            }

            private static bool IsTLSAllocatorNoise(string message)
            {
                if (message == null) return false;
                
                // Filtra as mensagens especificas do TLS Allocator da DLL nativa
                if (message.Contains("TLS Allocator ALLOC_TEMP_TLS"))
                    return true;
                if (message.Contains("ALLOC_TEMP_MAIN has unfreed allocations"))
                    return true;
                if (message.Contains("diag-temp-memory-leak-validation"))
                    return true;
                // Filtra as linhas individuais de "Allocation of X bytes at"
                if (message.Contains("Allocation of") && message.Contains("bytes at"))
                    return true;
                    
                return false;
            }
        }

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
        private static void Install()
        {
            Debug.unityLogger.logHandler = new SuppressiveLogHandler(Debug.unityLogger.logHandler);
            Debug.Log("[TLSAllocatorSuppressor] Filtro de warnings ALLOC_TEMP_TLS instalado. " +
                      "Warnings da DLL nativa sm64.dll serao suprimidos no Console.");
        }
    }
}
