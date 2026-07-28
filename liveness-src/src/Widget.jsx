import ReactDOM from 'react-dom/client';
import { FaceLivenessDetector } from '@aws-amplify/ui-react-liveness';
import '@aws-amplify/ui-react/styles.css';

let _root = null;

function LivenessChallenge({ sessionId, region, credentials, onSuccess, onError }) {
  return (
    <FaceLivenessDetector
      sessionId={sessionId}
      region={region}
      onAnalysisComplete={onSuccess}
      onError={(err) => {
        console.error('[AWS Liveness]', err);
        onError(err);
      }}
      config={{
        credentialProvider: async () => ({
          accessKeyId:     credentials.AccessKeyId,
          secretAccessKey: credentials.SecretAccessKey,
          sessionToken:    credentials.SessionToken,
        }),
      }}
    />
  );
}

export function mount(containerId, props) {
  const el = document.getElementById(containerId);
  if (!el) throw new Error('LivenessWidget: container not found: ' + containerId);
  _root = ReactDOM.createRoot(el);
  _root.render(<LivenessChallenge {...props} />);
}

export function unmount() {
  if (_root) { _root.unmount(); _root = null; }
}
